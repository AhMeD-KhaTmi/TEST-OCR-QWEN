import re

pattern = re.compile(
    r'^(?:(?:[1-9]|[1-4][0-9]|50)\s(?:[1-9]|[1-4][0-9]|50)|(?:[1-9]|[1-4][0-9]|50)[.\-][\d]{1,3}|[1-9]|[1-4][0-9]|50)$',
    re.VERBOSE | re.IGNORECASE
)

tests = [
    ('7 1', 'NOTE'),
    ('7 11', 'NOTE'),
    ('40 055', 'NUMBER'),
    ('829 417', 'NUMBER'),
    ('10 892', 'NUMBER'),
    ('7.1', 'NOTE'),
    ('7-11', 'NOTE'),
    ('1', 'NOTE'),
    ('50', 'NOTE'),
    ('51', 'NUMBER'),
]

print('Testing note pattern:')
print('-' * 50)
for val, expected in tests:
    matches = bool(pattern.match(val))
    result = 'NOTE' if matches else 'NUMBER'
    status = '✓' if result == expected else '✗'
    print(f'{status} {val:15} → {result:10} (expected {expected})')
