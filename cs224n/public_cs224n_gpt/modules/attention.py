import torch

from einops import rearrange
from torch import nn
import math
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    assert config.hidden_size % config.num_attention_heads == 0
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    b,h,t,d = key.size()

    ## calc original att
    att = (query @ key.transpose(-2,-1)) * (1.0 / math.sqrt(d)) # (b,h,t,t)
    ## apply mask on att (remember to set tensor on the same device)
    causal_mask = torch.tril(torch.ones(t,t,device=att.device)).view(1,1,t,t)
    att = att.masked_fill(causal_mask == 0 ,float('-inf'))
    att = att + attention_mask
    ## apply softmax on att
    att = F.softmax(att,dim=-1)
    ## apply dropout on att
    att = self.dropout(att)

    ## calc final output
    y = att @ value  # (b,h,t,d)
    ## reshape to original shape
    y = y.transpose(1,2).contiguous().reshape(b,t,h*d)
    return y

  def flash_attention(self, key, query, value, attention_mask):
    """
    Note that we implement the flash attention in pytorch way just to understand the algorithm.
    If we want to achive the best performance, we should use the flash attention package or implement 
    the code with CUDA programming, which may be more difficult.
    """
    batch_size, num_heads, seq_len, head_dim = key.size()

    scale = 1.0 / math.sqrt(head_dim) # scale for softmax

    block_size = 64 # flash attention block size
    o = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=key.device)
    
    # flash attention loop
    for i in range(0, seq_len, block_size):
      # outer loop for Q,O,m,l
      i_end = min(i + block_size, seq_len)
      q_i = query[:,:,i:i_end,:]
      o_i = torch.zeros_like(q_i)
      m_i = torch.full((batch_size, num_heads, i_end - i, 1), float('-inf'), device=key.device)
      l_i = torch.zeros((batch_size, num_heads, i_end - i, 1), device=key.device)
      for j in range(0,seq_len,block_size):
        # inner loop for K,V
        j_end = min(j + block_size, seq_len)
        k_j = key[:,:,j:j_end,:]
        v_j = value[:,:,j:j_end,:]
        
        # scaled dot product attention
        s_ij = torch.matmul(q_i,k_j.transpose(-2,-1)) * scale

        # apply attention mask
        i_idx = torch.arange(i,i_end,device=query.device).unsqueeze(-1)
        j_idx = torch.arange(j,j_end,device=query.device).unsqueeze(0)
        causal_mask = (i_idx >= j_idx).float() # (block_i,block_j)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # (1,1,block_i,block_j)

        s_ij = s_ij.masked_fill(causal_mask == 0,float('-inf'))
        if attention_mask is not None:
          mask_ij = attention_mask[:,:,:,j:j_end]
          mask_ij = mask_ij.expand(batch_size,num_heads,i_end-i,j_end-j)
          s_ij = s_ij + mask_ij
        
        # we do not apply dropout here because it will make flash attn more complicated

        # update m_i and l_i
        m_ij = torch.max(s_ij,dim=-1,keepdim=True)[0]
        m_i_new = torch.max(m_i,m_ij)

        exp_new = torch.exp(s_ij - m_i_new)
        exp_diff = torch.exp(m_i - m_i_new)

        l_i_new = exp_diff * l_i + torch.sum(exp_new,dim=-1,keepdim=True)
        o_i = exp_diff * o_i + torch.matmul(exp_new,v_j)
        m_i = m_i_new
        l_i = l_i_new

      o[:,:,i:i_end,:] = o_i / l_i
    
    o = o.transpose(1,2).contiguous().reshape(batch_size,seq_len,num_heads*head_dim)
    return o


  def flash_attention_real(self, key, query, value, attention_mask):
    """
    This is the real flash attention implementation.
    使用 flash-attn 库实现的高效注意力机制
    完全兼容原有 attention 函数的接口
    
    Args:
        key: [b, h, t, d]
        query: [b, h, t, d] 
        value: [b, h, t, d]
        attention_mask: [b, 1, 1, t]
    
    Returns:
        y: [b, t, h*d] - 与原函数完全相同的输出格式

    Note that when use flashattn,model tensor must be bf16 or fp16
    """
    try:
      from flash_attn import flash_attn_func
    except ImportError:
      raise ImportError("flash_attn is not installed. Please install it with `pip install flash-attn`.")
    
    b,h,t,d = key.size()
    # flash-attn 期望的输入格式: [batch, seq_len, num_heads, head_dim]
    # 我们的格式: [batch, num_heads, seq_len, head_dim]
    # 需要转换格式
    q = query.transpose(1, 2)  # [b, t, h, d]
    k = key.transpose(1, 2)    # [b, t, h, d]
    v = value.transpose(1, 2)  # [b, t, h, d]

    output = flash_attn_func(
        q, k, v,
        dropout_p=self.dropout.p if self.training and hasattr(self, 'dropout') else 0.0,
        causal=True,  # 对于 GPT 模型使用因果掩码
        window_size=(-1, -1),  # 不限制窗口大小
        alibi_slopes=None,
        return_attn_probs=False
    )  # 返回形状: [b, t, h, d]
    
    # 转换回原来的输出格式: [b, t, h*d]
    output = output.reshape(b, t, h * d)
    return output

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)

    # use flash attention
    # attn_value = self.flash_attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
