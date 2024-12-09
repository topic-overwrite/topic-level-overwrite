def deepspeed_init_distributed_mode():
    import deepspeed
    from deepspeed import get_accelerator
    import socket
    import torch
    import torch.distributed as dist
    import time
    import os
    if 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = global_rank % torch.cuda.device_count()

        job_id = os.environ['SLURM_JOBID']
        host_file = 'dist_url.' + job_id + '.txt'

        def find_free_port():
            s = socket.socket()
            s.bind(('', 0))
            return s.getsockname()[1]

        if global_rank == 0:
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            dist_url = 'tcp://{}:{}'.format(ip, port)
            with open(host_file, 'w') as f:
                f.write(dist_url)
        else:
            while not os.path.exists(host_file):
                time.sleep(1)
            with open(host_file, 'r') as f:
                dist_url = f.read()
    else:
        print('Not using distributed mode')
        return

    get_accelerator().set_device(local_rank)

    print('| distributed init (rank {}): {}'.format(
        global_rank, dist_url), flush=True)

    master_addr = dist_url[6:].split(':')[0].strip()
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    dist.init_process_group(
        backend='nccl', 
        init_method=dist_url,
        world_size=world_size, 
        rank=global_rank, 
    )
    deepspeed.init_distributed()
    dist.barrier()

    if global_rank == 0:
        for host_file in os.listdir('.'):
            if host_file.startswith('dist_url.'):
                os.remove(host_file)

    return dict(
        local_rank=local_rank, 
        global_rank=global_rank, 
        world_size=world_size, 
        master_addr=master_addr)


def torchrun_init_distributed_mode():
    import os
    import time
    import socket
    import torch
    import torch.distributed as dist
    if 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = global_rank % torch.cuda.device_count()

        job_id = os.environ['SLURM_JOBID']
        host_file = 'dist_url.' + job_id + '.txt'

        def find_free_port():
            s = socket.socket()
            s.bind(('', 0))
            return s.getsockname()[1]

        if global_rank == 0:
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            dist_url = 'tcp://{}:{}'.format(ip, port)
            with open(host_file, 'w') as f:
                f.write(dist_url)
        else:
            while not os.path.exists(host_file):
                time.sleep(1)
            with open(host_file, 'r') as f:
                dist_url = f.read()
    else:
        print('Not using distributed mode')
        return
    
    print('| distributed init (rank {}): {}'.format(
        global_rank, dist_url), flush=True)

    master_addr = dist_url[6:].split(':')[0].strip()
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    dist.init_process_group(
        backend='nccl', 
        init_method=dist_url,
        world_size=world_size, 
        rank=global_rank, 
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    dist.barrier()

    if global_rank == 0:
        for host_file in os.listdir('.'):
            if host_file.startswith('dist_url.'):
                os.remove(host_file)