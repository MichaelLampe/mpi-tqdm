import asyncio
from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, RLock, freeze_support

from tqdm import tqdm
from mpi4py import MPI
import threading

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
MAIN_THREAD = 0
PROGRESS_TAG = 234

LOOP_COUNT = 100000

rank_and_size = []
for i in range(0, SIZE):
    rank_and_size.append((i, LOOP_COUNT))

if MPI.COMM_WORLD.Get_rank() == 0:
    bars = {}
    for (position, total) in rank_and_size:
        bars[position] = tqdm(
            total=total,
            position=position,
            desc=f"Rank {position:03}",
            miniters=0

        )
        bars[position].update(0)

STOP_SIGNAL = False

async def render_all():
    """
    Refresh the progress bars every <Interval>
    """
    REFRESH_INTERVAL_IN_SECONDS = 1
    while not STOP_SIGNAL:
        for bar in bars.values():
            bar.display()
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)
        if STOP_SIGNAL:
            for bar in bars.values():
                bar.display()

async def other_rank_progress():
    """
    Receive messages from the other ranks, that indicate their progress so far
    """
    KEEP_LOOPING = True
    LAST_ITERATION = False

    MAX_ITERATIONS_WITHOUT_A_SHORT_WAIT = 1000
    ITERATIONS_SINCE_LAST_WAIT = 0

    while KEEP_LOOPING or not LAST_ITERATION:        
        # Check if there is a progress update to receive
        for rank in range(1, SIZE):
            request = COMM.irecv(source=rank, tag=PROGRESS_TAG)
            complete, result = request.test()
            if complete:
                if result:
                    n = result
                    bars[rank].update(n)

        # request = COMM.irecv(tag=PROGRESS_TAG)
        # complete, result = request.test()
        # if complete:
        #     if result:
        #         (rank, n) = result
        #         bars[rank].update(n)

        # If we are asked to stop and haven't received any results, let's just stop
        if STOP_SIGNAL:
            if result is None:
                LAST_ITERATION = True
                KEEP_LOOPING = False

        # If we haven't received any results in a while, let's just wait a bit
        if result is None:
            # We're waiting, so back at 0
            ITERATIONS_SINCE_LAST_WAIT = 0
            await asyncio.sleep(0.2)
        else:
            ITERATIONS_SINCE_LAST_WAIT += 1

            # If we have anything else using this thread and are iterating quickly (Not doing the 1 second wait),
            # we can starve any other tasks.  This is a hacky wait of just giving a slight enough pause that
            # something else can pickup the thread for a bit
            if ITERATIONS_SINCE_LAST_WAIT >= MAX_ITERATIONS_WITHOUT_A_SHORT_WAIT:
                ITERATIONS_SINCE_LAST_WAIT = 0
                await asyncio.sleep(0.000001)

def progress_updater(loop):
    asyncio.set_event_loop(loop)
    a = loop.create_task(other_rank_progress())
    b = loop.create_task(render_all())
    loop.run_until_complete(asyncio.wait([a, b]))

def update_progress(n=1):
    if MPI.COMM_WORLD.Get_rank() == 0:
        bars[0].update(n)
    else:
        COMM.isend(n, dest=MAIN_THREAD, tag=PROGRESS_TAG)

if __name__ == '__main__':   
    # Star the polling thread
    if MPI.COMM_WORLD.Get_rank() == 0:
        loop = asyncio.get_event_loop()
        t1 = threading.Thread(target=progress_updater, args=(loop,))
        t1.start()

    MPI.COMM_WORLD.barrier()
    
    # Setup poll loop
    for i in range(0, LOOP_COUNT):
        update_progress()

    MPI.COMM_WORLD.barrier()
    STOP_SIGNAL = True

