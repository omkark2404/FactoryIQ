import os
import json

root = 'data/raw/mvtec'
categories = ['bottle', 'screw', 'metal_nut', 'tile']
res = {}

for cat in categories:
    res[cat] = {
        'train_good': 0,
        'test_good': 0,
        'test_defect': 0,
        'defect_types': [],
        'ground_truth': 0
    }
    cat_dir = os.path.join(root, cat)
    if not os.path.exists(cat_dir):
        continue

    train_dir = os.path.join(cat_dir, 'train', 'good')
    if os.path.exists(train_dir):
        res[cat]['train_good'] = len([f for f in os.listdir(train_dir) if f.endswith('.png')])

    test_dir = os.path.join(cat_dir, 'test')
    if os.path.exists(test_dir):
        for sub in os.listdir(test_dir):
            sub_path = os.path.join(test_dir, sub)
            if not os.path.isdir(sub_path): continue
            count = len([f for f in os.listdir(sub_path) if f.endswith('.png')])
            if sub == 'good':
                res[cat]['test_good'] = count
            else:
                res[cat]['test_defect'] += count
                res[cat]['defect_types'].append(sub)

    gt_dir = os.path.join(cat_dir, 'ground_truth')
    if os.path.exists(gt_dir):
        for sub in os.listdir(gt_dir):
            sub_path = os.path.join(gt_dir, sub)
            if not os.path.isdir(sub_path): continue
            count = len([f for f in os.listdir(sub_path) if f.endswith('.png')])
            res[cat]['ground_truth'] += count

print(json.dumps(res, indent=2))
