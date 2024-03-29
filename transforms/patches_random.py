import torch


def _find_boundaries(x, y, w, h, image_shape, patch_size, min_overlap):
    max_h, min_h = max(h, patch_size[1]), min(h, patch_size[1])
    max_w, min_w = max(w, patch_size[0]), min(w, patch_size[0])

    min_y = y - max_h + min_overlap * min_h
    min_x = x - max_w + min_overlap * min_w
    max_y = y + h - int(min_overlap * min_h)
    max_x = x + w - int(min_overlap * min_w)

    min_y = int(max(min_y, 0))
    min_x = int(max(min_x, 0))
    max_y = int(max(min(max_y, image_shape[0] - patch_size[1] - 1), min_y))
    max_x = int(max(min(max_x, image_shape[1] - patch_size[0] - 1), min_x))

    return min_x, max_x, min_y, max_y

class RandomPatches(torch.nn.Module):
    def __init__(self, patch_size=(1024, 1024), min_overlap=0.9):
        super(RandomPatches, self).__init__()
        self.min_overlap = min_overlap
        self.patch_size = patch_size

    def forward(self, sample):
        image_tensor_list, item = sample['image_tensor_list'], sample['item']
        image_shape = image_tensor_list[-1].shape[1:3]

        abnorm_w = (item['maxx'] - item['minx']) / 2
        abnorm_x = int(abnorm_w + item['minx'])
        abnorm_h = (item['maxy'] - item['miny']) / 2
        abnorm_y = int(abnorm_h + item['miny'])

        min_x, max_x, min_y, max_y = _find_boundaries(abnorm_x, abnorm_y, abnorm_w, abnorm_h, image_shape, self.patch_size,
                                                      self.min_overlap)

        patch_y = torch.randint(min_y, max_y + 1, (1,))
        patch_x = torch.randint(min_x, max_x + 1, (1,))

        out_tensors = []
        for image_tensor in image_tensor_list:
            image_tensor_cropped = image_tensor[:, patch_y: patch_y + self.patch_size[1], patch_x: patch_x + self.patch_size[0]]
            out_tensors.append(image_tensor_cropped)

        sample = {'image_tensor_list': out_tensors, 'item': item}
        return sample

    def __repr__(self):
        detail = f"(patch_size={self.patch_size}, min_overlap={self.min_overlap})"
        return f"{self.__class__.__name__}{detail}"