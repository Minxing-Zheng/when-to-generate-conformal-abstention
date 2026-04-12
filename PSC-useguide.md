
  Last updated: 2026-03-23  
  
Purpose: this document is a working reference for future PSC/Bridges-2 tasks. It collects the key operational facts, common Slurm commands, known limits, and links to official PSC pages.  
  
Important rule: treat this document as a starting point, not as a live source of truth. PSC scheduling policies, QOS limits, partition limits, and queue conditions can change. Before making decisions that depend on current system state, log in and re-check with live commands.  
  
## Official Sources  
Enable searching when you cannot find enough information in this reference.
  
- Bridges-2 User Guide: <https://www.psc.edu/resources/bridges-2/user-guide/>  
- Bridges-2 FAQ: <https://www.psc.edu/resources/bridges-2/faq/>  
- Interactive sessions: <https://www.psc.edu/resources/bridges-2/user-guide/#interactive-sessions>  
- Batch jobs: <https://www.psc.edu/resources/bridges-2/user-guide/#batch-jobs>  
- Partitions: <https://www.psc.edu/resources/bridges-2/user-guide/#rm-partitions>  
- EM partition: <https://www.psc.edu/resources/bridges-2/user-guide/#em-partitions>  
- GPU partitions: <https://www.psc.edu/resources/bridges-2/user-guide/#gpu-partitions>  
- Job status / queue info: <https://www.psc.edu/resources/bridges-2/user-guide/#status-info>  
- OnDemand: <https://www.psc.edu/resources/bridges-2/user-guide/#ondemand>  
- File transfer: <https://www.psc.edu/resources/bridges-2/user-guide/#transferring-files>  
  
## Mental Model  
  
Bridges-2 has three main ways to run work:  
  
- OnDemand: web UI for interactive apps and job submission/tracking.  
- `interact`: command-line interactive Slurm allocation.  
- `sbatch`: command-line batch submission.  
  
Useful shorthand:  
  
- OnDemand interactive app ~= web entry point to an interactive job.  
- `interact` ~= shell entry point to an interactive job.  
- `sbatch` ~= non-interactive queued batch job.  
  
The user should not do real compute work on the login node. Login nodes are for setup, file management, environment work, and job submission.  
  
## Login and File Transfer  
  
### Login host  
  
Typical SSH login:  
  
```bash  
ssh PSC_USERNAME@bridges2.psc.edu  
```  
  
This lands on a login node, not a compute node.  
  
### Transfer host  
  
PSC documents file transfer through the data transfer nodes:  
  
```bash  
data.bridges2.psc.edu  
```  
  
Typical sync pattern:  
  
```bash  
rsync -avz ./myproject/ PSC_USERNAME@data.bridges2.psc.edu:/ocean/projects/GROUPNAME/PSC_USERNAME/myproject/  
```  
  
Typical small copy:  
  
```bash  
scp -r ./myproject PSC_USERNAME@data.bridges2.psc.edu:/ocean/projects/GROUPNAME/PSC_USERNAME/  
```  
  
Notes:  
  
- Home quota is 25 GB.  
- Larger working data should usually live under project storage such as `/ocean/projects/...`.  
- For large or many-file transfers, PSC recommends Globus.  
  
## Allocations and Accounts  
  
Before doing anything else, check what allocation and resource types the user actually has:  
  
```bash  
projects  
```  
  
This is critical because many common Slurm errors on Bridges-2 are caused by using the wrong account or a partition not covered by the current allocation.  
  
If the user has multiple allocations, explicitly set the charge account:  
  
Interactive:  
  
```bash  
interact -A CHARGE_ID ...  
```  
  
Batch:  
  
```bash  
#SBATCH -A CHARGE_ID  
```  
  
or  
  
```bash  
sbatch -A CHARGE_ID job.slurm  
```  
  
## `interact` vs `sbatch`  
  
### `interact`  
  
Use when:  
  
- debugging  
- testing commands  
- checking environment behavior  
- running short exploratory jobs  
  
Simple example:  
  
```bash  
interact  
```  
  
Useful explicit example:  
  
```bash  
interact -p RM-shared --ntasks-per-node=4 -t 01:00:00  
```  
  
Important documented behavior:  
  
- default partition: `RM-shared`  
- default cores: 1  
- default walltime: 60 minutes  
- maximum interactive time: 8 hours  
- inactive interactive jobs are logged out after 30 minutes idle  
- EM does not allow interactive sessions  
  
### `sbatch`  
  
Use when:  
  
- running training or longer jobs  
- leaving work in the queue/background  
- reproducing runs  
- avoiding staying attached to a shell  
  
Simple script:  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p RM-shared  
#SBATCH -t 01:00:00  
#SBATCH --ntasks-per-node=4  
#SBATCH -o slurm-%j.out  
  
hostname  
date  
python script.py  
```  
  
Submit:  
  
```bash  
sbatch job.slurm  
```  
  
Inspect:  
  
```bash  
squeue -u PSC_USERNAME  
```  
  
Cancel:  
  
```bash  
scancel JOBID  
```  
  
Important runtime semantics:  
  
- `#SBATCH -t ...` is a maximum allowed runtime, not a promise to occupy the node for the full duration  
- if the script finishes early, the job ends immediately  
- if the job exceeds the requested walltime, Slurm will typically terminate it  
  
### Practical distinction  
  
- `interact`: person stays online and types commands after the allocation starts.  
- `sbatch`: commands are written ahead of time and run later when scheduled.  
  
### Multi-step `sbatch` jobs  
  
One batch script can run multiple commands or multiple shell scripts in sequence. This is normal.  
  
Example:  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p RM-shared  
#SBATCH -t 01:00:00  
  
python preprocess.py  
python train.py  
python eval.py  
```  
  
or:  
  
```bash  
bash step1.sh  
bash step2.sh  
bash step3.sh  
```  
  
### Failure handling inside `sbatch`  
  
By default, a shell script does not always stop at the first failed command. If behavior matters, make it explicit.  
  
Fail fast:  
  
```bash  
set -e  
```  
  
This makes the job exit when a command returns a non-zero status.  
  
Allow one step to fail and continue:  
  
```bash  
python optional_step.py || true  
python next_step.py  
```  
  
More explicit form:  
  
```bash  
if ! python optional_step.py; then  
    echo "optional step failed, continuing"fi  
```  
  
Practical pattern:  
  
- use `set -e` for required steps  
- use `|| true` or an `if` block for optional post-processing or reporting steps  
  
### Parallel tasks inside one job  
  
It is possible to start multiple background processes inside one `sbatch` job, for example:  
  
```bash  
python a.py &  
python b.py &  
wait  
```  
  
But this should be done carefully because all processes will share the job's allocated resources. For most users, separate jobs or a Slurm job array is safer and easier to reason about.  
  
### Choosing between one `sbatch`, a job array, and multiple separate `sbatch` jobs  
  
Use one `sbatch` when:  
  
- the workflow is one pipeline  
- steps depend on each other  
- the work should run sequentially in one job  
  
Typical case:  
  
- preprocess -> train -> eval  
  
Use a job array when:  
  
- there are many similar tasks  
- the script is the same but the input, seed, or parameter changes  
- you want one submission to generate many related tasks  
  
Typical case:  
  
- many seeds  
- many hyperparameter variants  
- many file shards  
  
Use multiple separate `sbatch` jobs when:  
  
- tasks are independent  
- scripts differ materially  
- resource requests differ materially  
- each job should be managed, retried, or queued independently  
  
Typical case:  
  
- one 1-GPU task  
- one 4-GPU task  
- one preprocessing task  
- one evaluation task  
  
Rule of thumb:  
  
- one pipeline -> one `sbatch`  
- many similar independent tasks -> job array  
- many different independent tasks -> multiple `sbatch` jobs  
  
### Using one multi-GPU job to run several smaller tasks in parallel  
  
If a single `sbatch` job has already allocated a node with multiple GPUs, it is often better to launch several processes inside that same job instead of submitting more `sbatch` jobs.  
  
Typical reason:  
  
- one training run uses only 1 GPU  
- the allocated node has 4 or 8 GPUs  
- running tasks sequentially would leave some GPUs idle  
  
Basic pattern:  
  
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --seed 1 &  
CUDA_VISIBLE_DEVICES=1 python train.py --seed 2 &  
CUDA_VISIBLE_DEVICES=2 python train.py --seed 3 &  
CUDA_VISIBLE_DEVICES=3 python train.py --seed 4 &  
wait  
```  
  
Notes:  
  
- `&` launches the process in the background  
- `wait` keeps the batch job alive until all launched processes finish  
- `CUDA_VISIBLE_DEVICES` prevents processes from fighting over the same GPU  
  
This is often the right choice when:  
  
- the node is already allocated  
- tasks are independent  
- each task uses only part of the allocated GPUs  
  
But it still requires planning for:  
  
- CPU sharing  
- memory sharing  
- log separation  
- failure handling  
  
If each process needs different resources or should be queued independently, use separate jobs or a job array instead.  
  
## Partitions and What They Mean  
  
### RM family  
  
- `RM`: whole regular-memory nodes  
- `RM-shared`: part of one regular-memory node  
- `RM-512`: whole 512 GB nodes  
  
Publicly documented RM summary:  
  
- RM node RAM: 256 GB  
- RM-shared node RAM: 256 GB shared node  
- RM-512 node RAM: 512 GB  
- RM / RM-shared / RM-512 documented max walltime in the user guide table: 72 hours  
  
Important caveat:  
  
- PSC FAQ says limits can change and recommends checking live QOS values.  
- The FAQ example shows 2 days for `rmpartition` and `rm512partition`.  
- Therefore, do not trust static walltime numbers blindly. Re-check on the live system.  
  
### EM  
  
- Extreme memory nodes  
- 4 TB memory per node  
- 96 cores per node  
- one node maximum  
- cores must be requested in multiples of 24  
- no interactive jobs  
- no OnDemand use for EM according to the user guide  
- documented max walltime: 120 hours / 5 days  
  
### GPU family  
  
- `GPU`: whole GPU nodes  
- `GPU-shared`: part of one GPU node  
  
Publicly documented GPU partition summary:  
  
- default runtime: 1 hour  
- max runtime: 48 hours  
- `GPU-shared` maximum GPUs per job: 4  
- `GPU` can use whole nodes and multiple nodes  
  
Important distinction:  
  
- `GPU-shared` is for up to 4 GPUs from one shared node  
- `GPU` is for whole nodes, typically 8 GPUs per standard node  
  
## GPU Inventory  
  
According to the Bridges-2 user guide as checked on 2026-03-23:  
  
- `h100-80`: 10 nodes, 8 H100 GPUs per node, 80 GB GPU memory each, 2 TB RAM per node  
- `l40s-48`: 3 nodes, 8 L40S GPUs per node, 48 GB GPU memory each, 1 TB RAM per node  
- `v100-32`: 24 standard nodes, 8 V100 GPUs per node, 32 GB GPU memory each, 512 GB RAM per node  
- `v100-16`: 9 nodes, 8 V100 GPUs per node, 16 GB GPU memory each, 192 GB RAM per node  
- DGX-2: 1 node with 16 `v100-32` GPUs, 1.5 TB RAM  
  
Approximate totals:  
  
- H100 GPUs: 80  
- L40S GPUs: 24  
- V100-32 GPUs on standard nodes: 192  
- V100-16 GPUs: 72  
- DGX-2 V100-32 GPUs: 16  
- Total V100 GPUs: 280  
  
## Common GPU Request Patterns  
  
### 4 H100 GPUs  
  
Typically:  
  
- partition: `GPU-shared`  
- resource request: part of one H100 node  
  
Example:  
  
```bash  
sbatch -p GPU-shared --gpus=h100-80:4 -t 04:00:00 job.slurm  
```  
  
### 8 H100 GPUs  
  
Typically:  
  
- partition: `GPU`  
- resource request: one whole H100 node  
  
Example:  
  
```bash  
sbatch -p GPU --gpus=h100-80:8 -t 04:00:00 job.slurm  
```  
  
### Interactive GPU example  
  
Interactive uses `--gres` in PSC examples:  
  
```bash  
interact -p GPU-shared --gres=gpu:h100-80:1 -t 00:30:00  
```  
  
### Batch GPU example  
  
Batch uses `--gpus` in PSC examples:  
  
```bash  
sbatch -p GPU --gpus=h100-80:8 -t 02:00:00 job.slurm  
```  
  
Note: the user guide has some inconsistency in examples between `--gres` and `--gpus`. The most stable reading is:  
  
- interactive: prefer `--gres=gpu:type:n`  
- batch: prefer `--gpus=type:n`  
  
## Service Unit Notes  
  
PSC defines GPU usage in gpu-hours.  
  
Documented rates:  
  
- v100: 1 GPU-hour = 1 SU  
- l40s: 1 GPU-hour = 1 SU  
- h100: 1 GPU-hour = 2 SU  
  
Examples:  
  
- 1 full H100 node for 1 hour = 8 GPU-hours = 16 SUs  
- 4 H100 GPUs for 48 hours = 192 GPU-hours = 384 SUs  
  
## Queueing and Priority Policy  
  
What PSC publicly states in the user guide:  
  
- all partitions use FIFO scheduling  
- if the top job will not fit, Slurm tries to schedule the next job in the partition  
- the scheduler follows policies to ensure one user does not dominate the machine  
- there are limits to the number of nodes and cores a user can simultaneously use  
- scheduling policies are under review and can change  
  
This implies:  
  
- there is a FIFO base policy  
- there is backfill behavior  
- there are fairness and user-cap mechanisms  
  
Inference from PSC-provided tools:  
  
- PSC exposes `slurm-tool prio` and `slurm-tool shares`  
- that strongly suggests fair-share and priority components matter in practice  
  
Important wording:  
  
- the above fair-share interpretation is an inference from available tools, not a direct public formula from PSC documentation  
  
## Known Limit / QOS Policy Notes  
  
PSC FAQ explicitly documents several common policy-related errors:  
  
- `Invalid qos specification`  
- `Invalid account or account/partition combination specified`  
- `QOSMaxCpuPerJobLimit`  
- `QOSMaxWallDurationPerJobLimit`  
- generic `Job violates accounting/QOS policy (job submit limit, user’s size and/or time limits)`  
  
What these usually mean:  
  
- wrong account for the chosen partition  
- asking for a resource type the allocation does not include  
- too many cores / GPUs / nodes for that partition  
- too much walltime  
- possibly job count or submit count limits  
  
Examples PSC explicitly states:  
  
- `GPU-shared` maximum is 4 GPUs  
- multiple GPU nodes must use `GPU`, not `GPU-shared`  
- `RM-shared` max cores is half a node, 64 cores  
  
### OnDemand VS Code session limit  
  
Observed by the user:  
  
- OnDemand VS Code failed when trying to open a second session and reported a max limit style message  
  
What is public vs not public:  
  
- I did not find a PSC public page that explicitly says "VS Code sessions per user = 1"  
- therefore this should not be treated as a documented platform-wide constant without live verification  
  
Best interpretation:  
  
- the user likely hit a QOS, interactive app, or per-user limit  
- the exact limit should be checked on the live system or with PSC support if it matters operationally  
  
## Commands to Check Live Limits  
  
These are the most important commands when a task depends on current PSC policy instead of static docs.  
  
### Current max walltime by partition QOS  
  
PSC FAQ explicitly recommends:  
  
```bash  
sacctmgr show qos format=name%15,maxwall | grep partition  
```  
  
### Useful extra live checks  
  
These are standard Slurm/accounting queries that may be available on Bridges-2:  
  
```bash  
sacctmgr show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU,GrpTRES,GrpJobs  
sacctmgr show assoc user=$USER format=User,Account,QOS,MaxJobs,MaxSubmit,GrpTRES,GrpJobs  
```  
  
If permission or output differs, fall back to PSC tools and `squeue`.  
  
## Core Operational Commands  
  
### Allocation and account  
  
```bash  
projects  
```  
  
### Start interactive CPU session  
  
```bash  
interact -p RM-shared --ntasks-per-node=4 -t 01:00:00  
```  
  
### Start interactive GPU session  
  
```bash  
interact -p GPU-shared --gres=gpu:h100-80:1 -t 00:30:00  
```  
  
### Submit batch script  
  
```bash  
sbatch job.slurm  
```  
  
### Submit batch script with explicit account  
  
```bash  
sbatch -A CHARGE_ID job.slurm  
```  
  
### Check own jobs  
  
```bash  
squeue -u $USER  
```  
  
### Check own jobs with more detail  
  
```bash  
squeue -l -u $USER  
```  
  
This can show:  
  
- requested time  
- elapsed time  
- requested nodes  
- node assignment  
- state  
- waiting reason  
  
### Cancel job  
  
```bash  
scancel JOBID  
```  
  
### Check completed job details  
  
```bash  
sacct -X -j JOBID -S 032326 --format=JobID,Partition,Account,State,ExitCode,Elapsed,MaxRSS,AllocCPUs  
```  
  
Useful fields from PSC docs:  
  
- `JobID`  
- `Partition`  
- `Account`  
- `ExitCode`  
- `State`  
- `Start`  
- `End`  
- `Elapsed`  
- `NodeList`  
- `NNodes`  
- `MaxRSS`  
- `AllocCPUs`  
  
### PSC helper commands  
  
Show queue and summaries:  
  
```bash  
slurm-tool -h  
slurm-tool queue  
slurm-tool quick  
slurm-tool full  
slurm-tool prio short  
slurm-tool shares  
slurm-tool partitions  
slurm-tool nodes  
slurm-tool cpus  
slurm-tool cpus queue  
```  
  
Show jobs for user/account/partition:  
  
```bash  
showuserjobs -u $USER  
showuserjobs -u $USER -r  
showuserjobs -p GPU -r  
showuserjobs -p GPU-shared -r  
showuserjobs -a CHARGE_ID -r  
```  
  
Node state:  
  
```bash  
sinfo  
sinfo -p GPU  
sinfo -p GPU-shared  
```  
  
Completed job accounting with PSC helper:  
  
```bash  
/opt/packages/allocations/bin/job_info JOBID  
/opt/packages/allocations/bin/job_info --slurm JOBID  
```  
  
## H100 Queue Investigation Workflow  
  
When someone asks "how long will 4 H100 or 8 H100 wait", do not give a fake fixed ETA. Check live state first.  
  
Recommended workflow:  
  
1. Check whether the account actually has GPU access.  
  
```bash  
projects  
```  
  
2. Check current H100-related queues.  
  
```bash  
squeue -p GPU-shared -l  
squeue -p GPU -l  
showuserjobs -p GPU-shared -r  
showuserjobs -p GPU -r  
```  
  
3. Check node availability.  
  
```bash  
sinfo -p GPU  
sinfo -p GPU-shared  
```  
  
4. Check priority and fair-share clues.  
  
```bash  
slurm-tool prio short  
slurm-tool shares  
```  
  
5. Interpret conservatively.  
  
General qualitative guidance:  
  
- 4 H100 on `GPU-shared` is usually easier to place than 8 H100 on `GPU`  
- shorter walltime is usually easier to backfill than longer walltime  
- actual wait can vary from quick start to many hours depending on current demand  
  
Do not claim a numeric wait estimate unless it is based on current live queue evidence.  
  
## OnDemand Notes  
  
PSC documents that OnDemand can:  
  
- manage files  
- submit and track jobs  
- see job output  
- check queue status  
  
Therefore:  
  
- basic queue and job visibility is available in OnDemand  
- shell access is still better for detailed diagnosis, especially waiting reasons, accounting, and priority inspection  
  
Practical guidance:  
  
- use OnDemand for convenience  
- use shell for detailed queue debugging and system inspection  
  
## Minimal Templates  
  
### Minimal CPU batch script  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p RM-shared  
#SBATCH -t 01:00:00  
#SBATCH --ntasks-per-node=4  
#SBATCH -o slurm-%j.out  
  
module load anaconda3  
python script.py  
```  
  
### Minimal sequential batch script with fail-fast behavior  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p RM-shared  
#SBATCH -t 02:00:00  
#SBATCH --ntasks-per-node=4  
#SBATCH -o slurm-%j.out  
  
set -e  
  
python preprocess.py  
python train.py  
python eval.py  
```  
  
### Minimal batch script where one later step may fail  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p RM-shared  
#SBATCH -t 02:00:00  
#SBATCH --ntasks-per-node=4  
#SBATCH -o slurm-%j.out  
  
set -e  
  
python preprocess.py  
python train.py  
python eval.py || echo "eval failed, continuing"  
```  
  
### Minimal multi-GPU batch script that runs several 1-GPU tasks in parallel  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p GPU-shared  
#SBATCH -t 04:00:00  
#SBATCH --gpus=h100-80:4  
#SBATCH --ntasks-per-node=16  
#SBATCH -o slurm-%j.out  
  
set -e  
  
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train.py --seed 1 &  
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python train.py --seed 2 &  
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train.py --seed 3 &  
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 python train.py --seed 4 &  
  
wait  
```  
  
### Minimal 4x H100 batch script  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p GPU-shared  
#SBATCH -t 04:00:00  
#SBATCH --gpus=h100-80:4  
#SBATCH -o slurm-%j.out  
  
nvidia-smi  
python train.py  
```  
  
### Minimal 8x H100 batch script  
  
```bash  
#!/bin/bash  
#SBATCH -A CHARGE_ID  
#SBATCH -p GPU  
#SBATCH -t 04:00:00  
#SBATCH --gpus=h100-80:8  
#SBATCH -o slurm-%j.out  
  
nvidia-smi  
python train.py  
```  
  
## Things To Re-Verify Before Any Serious PSC Task  
  
- current partition walltime limits  
- current QOS limits  
- whether the user's account includes GPU access  
- whether the correct charge account is being used  
- current queue congestion  
- current node availability  
- whether an observed OnDemand limit is an app limit, QOS limit, or account limit  
  
## Quick Summary  
  
- OnDemand interactive apps and `interact` are both interactive job models.  
- `sbatch` is the standard way to run longer unattended jobs.  
- PSC publicly states FIFO scheduling with additional policies to prevent one user from dominating the system.  
- `GPU-shared` is up to 4 GPUs on one shared node.  
- `GPU` is for whole GPU nodes.  
- H100 inventory is documented as 10 nodes x 8 GPUs.  
- L40S inventory is documented as 3 nodes x 8 GPUs.  
- V100 inventory is documented as 24 `v100-32` nodes, 9 `v100-16` nodes, plus 1 DGX-2 with 16 `v100-32`.  
- Many apparent "max limit" issues are really QOS/accounting/policy limits, not necessarily a fixed product-level session cap.
  
  
