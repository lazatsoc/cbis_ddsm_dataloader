import torch

from transforms.patches_random import _find_boundaries


def normal_patch_transform_wrapper(other_tranform, probability=0.5, patch_size=(1024, 1024), min_breast_overlap=0.5, max_abnorm_overlap=0.1, max_tries=5):
    def perform(sample):
        choice = torch.randint(0, 100, (1,))
        if choice >= probability * 100:
            return other_tranform(sample)
        else:
            image_tensor_list, item = sample['image_tensor_list'], sample['item']
            image_shape = image_tensor_list[0].shape

            abnorm_w = (item['maxx'] - item['minx']) / 2
            abnorm_x = int(abnorm_w + item['minx'])
            abnorm_h = (item['maxy'] - item['miny']) / 2
            abnorm_y = int(abnorm_h + item['miny'])

            abnorm_min_x, abnorm_max_x, abnorm_min_y, abnorm_max_y = _find_boundaries(abnorm_x, abnorm_y,
                                                                                      abnorm_w, abnorm_h,
                                                                                      image_shape, patch_size,
                                                                                      1 - max_abnorm_overlap)

            breast_w = (item['breast_maxx'] - item['breast_minx']) / 2
            breast_x = int(breast_w + item['breast_minx'])
            breast_h = (item['breast_maxy'] - item['breast_miny']) / 2
            breast_y = int(breast_h + item['breast_miny'])

            breast_min_x, breast_max_x, breast_min_y, breast_max_y = _find_boundaries(breast_x, breast_y,
                                                                                      breast_w, breast_h,
                                                                                      image_shape, patch_size,
                                                                                      min_breast_overlap)
            counter = 0
            while (True):
                patch_y = torch.randint(breast_min_y, breast_max_y, (1,))
                patch_x = torch.randint(breast_min_x, breast_max_x, (1,))

                if (patch_x < abnorm_min_x or patch_x > abnorm_max_x) and (
                        patch_y < abnorm_min_y or patch_y > abnorm_max_y):
                    break
                counter += 1
                if counter == 5:
                    print('Giving up')
                    return other_tranform(sample)

            out_tensors = []
            for image_tensor in image_tensor_list:
                image_tensor = image_tensor[patch_y: patch_y + patch_size[1], patch_x: patch_x + patch_size[0]]
                out_tensors.append(image_tensor)

            item['pathology'] = 'NORMAL'
            sample = {'image_tensor_list': out_tensors, 'item': item}
            return sample
        pass

    return perform
