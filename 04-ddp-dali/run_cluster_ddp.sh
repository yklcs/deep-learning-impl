#!/bin/sh

#SBATCH -J  CSED490F_W4           # Job name
#SBATCH -o  slurm_logs/%j.out    # Name of stdout output file (%j expands to %jobId)


#### Select  GPU
#SBATCH -p titanxp       # queue  name  or  partiton
#SBATCH   --gres=gpu:2          # gpus per node

##  node 지정하기
#SBATCH   --nodes=1              # the number of nodes 
#SBATCH   --ntasks-per-node=1
#SBATCH   --cpus-per-task=6

## Time limit
#SBATCH --time 01:00:00

# Set your name and home path
# For REPO_DIR, we recommend you to use same name as origin git repository name,
# but you can freely use new REPO name and revise repository path.
YOUR_NAME="your_name" # ex: honggildong
YOUR_HOME_PATH="your_home_path" # ex: /home/honggildong
REPO_DIR=CSED490F_DDP_DALI_Training
DATASET_DIR="/home/dataset"

DIRECTORY_PATH=${YOUR_HOME_PATH}/${REPO_DIR}
DOCKER_NAME=${YOUR_NAME}_CSED490F_DDP_DALI_Training_container
DOCKER_IMAGE=25fallcsed490f/cluster:week4
## function to init cleanup file
function cleanup {
    ## change permission in docker container created files
    docker exec ${DOCKER_NAME} bash -c "chmod -R 777 /HOME"
    ## docker container stop
    docker stop ${DOCKER_NAME}
    ## docker container remove
    docker rm -f ${DOCKER_NAME}
    exit
}

## Set Trap -> Slurm에서 SIGTERM signal을 사용
trap cleanup 0

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo $DOCKER_NAME

## download docker image
docker pull ${DOCKER_IMAGE}

## get GPU UUID
echo "[System] UUID GPU List"
UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"

## docker container background
echo "[System] create docker container..."
docker run -itd --gpus ${GPULIST} --name ${DOCKER_NAME} --ipc=host --shm-size=8G \
    -v ${DIRECTORY_PATH}:/HOME \
    -v ${DATASET_DIR}:/DATA \
    ${DOCKER_IMAGE}
echo "[System] docker container created"

echo "[System] initialize docker container..."
docker exec ${DOCKER_NAME} bash -c "cd /HOME && bash scripts/init.sh"
echo "[System] initialize docker container finished"

echo "[System] run train_cifar.py with 'run_ddp.sh'..."
docker exec ${DOCKER_NAME} bash -c "cd /HOME && bash scripts/run_ddp.sh"
echo "[System] train ended"

## Stop docker container
docker stop ${DOCKER_NAME}

## Remove docker container
docker rm -f ${DOCKER_NAME}

## Print slurm execution information
date
squeue  --job  $SLURM_JOBID

echo  "##### END #####"
