import os
from string import Template

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=str, required=True,
                    help='Language pairs, e.g. en2vi,ha2en,uz2en,hu2en')
parser.add_argument('--data-dir', type=str, required=True,
                    help='path to data dir')
parser.add_argument('--dump-dir', type=str, required=True,
                    help='path to dump dir')
parser.add_argument('-c', type=str, required=True,
                    help='config name')
parser.add_argument('-q', type=str, required=True, help='queue name')
parser.add_argument('-w', type=str, required=True, help='working dir')
parser.add_argument('-n', type=str, required=True, help='name')
args = parser.parse_args()

pairs = args.pairs
queue = args.q
config = args.c
working_dir = os.path.abspath(args.w)
name = args.n
drhu_queues = ['qa-xp-00{}'.format(i) for i in range(1, 9)]
gpu_card = 1  # if queue in drhu_queues else 1
template = Template("""#!/bin/csh

#$$ -M tnguye28@nd.edu
#$$ -m abe
#$$ -q gpu@$queue
#$$ -l gpu_card=$gpu_card
#$$ -N $name         # Specify job name

module load python/3.6.4

cd $working_dir

mkdir -p $working_dir/dump
touch $working_dir/dump/DEBUG.log
fsync -d 30 $working_dir/dump/DEBUG.log &

python3 main.py --mode train --data-dir ./data --dump-dir ./dump --pairs $pairs --config $config
""")


job_script = template.substitute(dict(queue=queue, gpu_card=gpu_card, name=name, working_dir=working_dir, pairs=pairs, config=config))
job_script_path = os.path.join(working_dir, 'train.job')
open(job_script_path, 'w').close()
with open(job_script_path, 'w') as f:
    f.write(job_script)
