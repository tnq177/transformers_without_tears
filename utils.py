from datetime import timedelta
import torch


def format_seconds(seconds):
    return str(timedelta(seconds=seconds))


def get_positional_encoding(dim, sentence_length):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    div_term = -(torch.arange(end=float(dim), device=device) // 2) * 2.0 / dim
    div_term = torch.pow(10000.0, div_term).reshape(1, -1)
    pos = torch.arange(end=float(sentence_length), device=device).reshape(-1, 1)
    encoded_vec = torch.matmul(pos, div_term)
    encoded_vec[:, 0::2] = torch.sin(encoded_vec[:, 0::2])
    encoded_vec[:, 1::2] = torch.cos(encoded_vec[:, 1::2])

    return encoded_vec.reshape([sentence_length, dim])
