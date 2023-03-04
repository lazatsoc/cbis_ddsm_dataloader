import torch


def centered_patch_transform(patch_size=(1024, 1024)):
    def perform(sample):
        image_tensor_list, item = sample['image_tensor_list'], sample['item']
        image_shape = image_tensor_list[0].shape

        cx = item['cx']
        cy = item['cy']

        minx_naive = int(cx - patch_size[0] / 2)
        minx = max((0, minx_naive))
        dx1 = minx_naive - minx

        maxx_naive = int(cx + patch_size[0] / 2)
        maxx = min((maxx_naive, image_shape[1]))
        dx2 = maxx - maxx_naive

        if dx1 != 0 and dx2 != 0:
            print('Warning: patch size bigger than image x-dimension. Please select a smaller patch size.')
        else:
            minx -= dx2
            maxx -= dx1

        miny_naive = int(cy - patch_size[1] / 2)
        miny = max((0, miny_naive))
        dy1 = miny_naive - miny

        maxy_naive = int(cy + patch_size[1] / 2)
        maxy = min((maxy_naive, image_shape[0]))
        dy2 = maxy - maxy_naive

        if dy1 != 0 and dy2 != 0:
            print('Warning: patch size bigger than image y-dimension. Please select a smaller patch size.')
        else:
            miny -= dy2
            maxy -= dy1

        out_tensors = []
        for image_tensor in image_tensor_list:
            image_tensor = image_tensor[miny: maxy, minx: maxx]
            out_tensors.append(image_tensor)

        return out_tensors, item

    return perform


def random_patch_transform(patch_size=(1024, 1024), min_overlap = 0.9):
    def perform(sample):
        image_tensor_list, item = sample['image_tensor_list'], sample['item']
        image_shape = image_tensor_list[0].shape

        abnorm_w = (item['maxx'] - item['minx']) / 2
        abnorm_x = int(abnorm_w + item['minx'])
        abnorm_h = (item['maxy'] - item['miny']) / 2
        abnorm_y = int(abnorm_h + item['miny'])

        max_h, min_h = max(abnorm_h, patch_size[1]), min(abnorm_h, patch_size[1])
        max_w, min_w = max(abnorm_w, patch_size[0]), min(abnorm_w, patch_size[0])

        min_y = abnorm_y - max_h + min_overlap * min_h
        min_x = abnorm_x - max_w + min_overlap * min_w
        max_y = abnorm_y + abnorm_h - int(min_overlap * min_h)
        max_x = abnorm_x + abnorm_w - int(min_overlap * min_w)

        min_y = int(max(min_y, 0))
        min_x = int(max(min_x, 0))
        max_y = int(max(min(max_y, image_shape[0] - patch_size[1] - 1), min_y))
        max_x = int(max(min(max_x, image_shape[1] - patch_size[0] - 1), min_x))

        patch_y = torch.randint(min_y, max_y + 1, (1,))
        patch_x = torch.randint(min_x, max_x + 1, (1,))

        out_tensors = []
        for image_tensor in image_tensor_list:
            image_tensor = image_tensor[patch_y: patch_y+patch_size[1], patch_x: patch_x+patch_size[0]]
            out_tensors.append(image_tensor)

        sample = {'image_tensor_list': out_tensors, 'item': item}
        return sample
    return perform


def normal_transform_wrapper(other_tranform, probability=0.5, patch_size=(1024, 1024), max_overlap=0.1):
    def perform(sample):
        choice = torch.randint(0, 100, (1,))
        if choice >= probability * 100:
            return other_tranform(sample)
        else:
            image_tensor_list, item = sample['image_tensor_list'], sample['item']
            image_shape = image_tensor_list[0].shape

            min_overlap = 1 - max_overlap

            abnorm_w = (item['maxx'] - item['minx']) / 2
            abnorm_x = int(abnorm_w + item['minx'])
            abnorm_h = (item['maxy'] - item['miny']) / 2
            abnorm_y = int(abnorm_h + item['miny'])

            abnorm_max_h, abnorm_min_h = max(abnorm_h, patch_size[1]), min(abnorm_h, patch_size[1])
            abnorm_max_w, abnorm_min_w = max(abnorm_w, patch_size[0]), min(abnorm_w, patch_size[0])

            abnorm_min_y = abnorm_y - abnorm_max_h + min_overlap * abnorm_min_h
            abnorm_min_x = abnorm_x - abnorm_max_w + min_overlap * abnorm_min_w
            abnorm_max_y = abnorm_y + abnorm_h - int(min_overlap * abnorm_min_h)
            abnorm_max_x = abnorm_x + abnorm_w - int(min_overlap * abnorm_min_w)

            abnorm_min_y = int(max(abnorm_min_y, 0))
            abnorm_min_x = int(max(abnorm_min_x, 0))
            abnorm_max_y = int(max(min(abnorm_max_y, image_shape[0] - patch_size[1] - 1), abnorm_min_y))
            abnorm_max_x = int(max(min(abnorm_max_x, image_shape[1] - patch_size[0] - 1), abnorm_min_x))

            while (True):
                patch_y = torch.randint(item['breast_miny'], item['breast_maxy'] - patch_size[1], (1,))
                patch_x = torch.randint(item['breast_minx'], item['breast_maxx'] - patch_size[0], (1,))

                if (patch_x < abnorm_min_x or patch_x > abnorm_max_x) and (
                        patch_y < abnorm_min_y or patch_y > abnorm_max_y):
                    break

            out_tensors = []
            for image_tensor in image_tensor_list:
                image_tensor = image_tensor[patch_y: patch_y + patch_size[1], patch_x: patch_x + patch_size[0]]
                out_tensors.append(image_tensor)

            item['pathology'] = 'NORMAL'
            sample = {'image_tensor_list': out_tensors, 'item': item}
            return sample
        pass

    return perform