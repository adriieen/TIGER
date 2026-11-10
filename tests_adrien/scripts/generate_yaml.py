splits = [
    'low_frequency_narrow_split',
    'normal_split',
    'even_split',
    'low_f0_split',
    'high_f0_split',
    'f1_split',
    'f2_split',
    'f3_split'
]

with open('TIGER/configs/tiger-tiny.yml', 'r') as f:
    base_content = f.read()

for split in splits:
    # 1. Add frequency_split to audionet_config
    new_content = base_content.replace(
        '    num_sources: 2\n',
        f'    num_sources: 2\n    frequency_split: {split}\n'
    )
    # 2. Update exp_name for distinct logging and checkpoints
    new_content = new_content.replace(
        'exp_name: TIGER-EchoSet-tiny',
        f'exp_name: TIGER-EchoSet-tiny-{split}'
    )
    filename = f'TIGER/configs/tiger-tiny-{split}.yml'
    with open(filename, 'w') as f:
        f.write(new_content)
    print(f'Created {filename}')
