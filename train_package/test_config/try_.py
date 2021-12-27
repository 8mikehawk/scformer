import yaml
import sys

f = open(sys.argv[1])
config = yaml.safe_load(f)

print(config)
print(sys.argv)
