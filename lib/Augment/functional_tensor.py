import torch
from torch import Tensor

def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return Tensor.dim(x) >= 2

def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def erase(img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace: bool = False) -> Tensor:
    _assert_image_tensor(img)
    if not inplace:
        img = img.clone()

    img[..., i: i + h, j: j + w] = v
    return img