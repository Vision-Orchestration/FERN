import re
from pathlib import Path

root = Path('C:/fern/FERN_V2')

skel_dir = root / 'data' / 'skeletons' / 'front'
test_dir = root / 'data' / 'skeletons' / 'front_test'
aug_dir = root / 'data' / 'skeletons' / 'front_aug'
aug_test_dir = root / 'data' / 'skeletons' / 'front_aug_test'

train_subjects = set()
for f in skel_dir.glob('*.csv'):
    m = re.match(r'^(p\d{2})', f.stem)
    if m: train_subjects.add(m.group(1))

test_subjects = set()
for f in test_dir.glob('*.csv'):
    m = re.match(r'^(p\d{2})', f.stem)
    if m: test_subjects.add(m.group(1))

train_csv = len(list(skel_dir.glob('*.csv')))
test_csv = len(list(test_dir.glob('*.csv')))
aug_csv = len(list(aug_dir.glob('*.csv')))
aug_test_csv = len(list(aug_test_dir.glob('*.csv')))

print(f'Training subjects: {len(train_subjects)}')
print(f'Held-out subjects: {len(test_subjects)} -> {sorted(test_subjects)}')
print(f'Training originals: {train_csv} files')
print(f'Held-out originals: {test_csv} files')
print(f'Training augmented: {aug_csv} files')
print(f'Held-out augmented: {aug_test_csv} files')
