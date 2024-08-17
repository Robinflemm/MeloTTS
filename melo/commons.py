import math
import torch
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):
    """Initialize the weights of convolutional layers with a normal distribution."""
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    """Calculate the padding size for a convolutional layer given the kernel size and dilation."""
    return (kernel_size * dilation - dilation) // 2

def convert_pad_shape(pad_shape):
    """Convert padding shape from a list of tuples to a flattened list for F.pad."""
    return [item for sublist in pad_shape[::-1] for item in sublist]

def intersperse(lst, item):
    """Interleave a given item between elements of the list."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """Compute the KL divergence between two distributions P and Q."""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl

def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protected from overflow."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))

def rand_gumbel_like(x):
    """Generate Gumbel noise with the same shape as input tensor x."""
    return rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)

def slice_segments(x, ids_str, segment_size=4):
    """Slice the input tensor into segments based on starting indices."""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        ret[i] = x[i, :, idx_str:idx_str + segment_size]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """Randomly slice segments from the input tensor x."""
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).long()
    return slice_segments(x, ids_str, segment_size), ids_str

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Generate a timing signal for positional encoding."""
    position = torch.arange(length, dtype=torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=0)
    signal = F.pad(signal, [0, 0, 0, channels % 2]).view(1, channels, length)
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Add a timing signal to the input tensor for positional encoding."""
    _, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    """Concatenate a timing signal with the input tensor along the specified axis."""
    _, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], dim=axis)

def subsequent_mask(length):
    """Create a mask for subsequent positions in the sequence to prevent attention."""
    return torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """Fused operation of addition, tanh, sigmoid, and multiplication."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act

def shift_1d(x):
    """Shift the input tensor by one position along the last dimension."""
    return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]

def sequence_mask(length, max_length=None):
    """Generate a mask for sequences of different lengths."""
    if max_length is None:
        max_length = length.max()
    return torch.arange(max_length, dtype=length.dtype, device=length.device).unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):
    """Generate a path based on duration and mask tensors."""
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask

def clip_grad_value_(parameters, clip_value, norm_type=2):
    """Clip the gradients of the parameters by a given value."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    if clip_value is not None:
        clip_value = float(clip_value)

    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    return total_norm ** (1.0 / norm_type)
