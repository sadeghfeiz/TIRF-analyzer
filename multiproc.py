import time
import multiprocessing as mp
import numpy as np
from imgProc import *
def slow_calculataion(imgMatrix, d):
    d = findPeaks(d) # Find the peaks in the reference image
    d = intensityTracker(imgMatrix, d) # Extract the spots' intensity over different frames
    return (d)
def compute_parallel(imgMatrix, d, nmbr_processes):
    # --- initialize the multiprocessing ----
    inQ = mp.Queue()
    outQ = mp.Queue()
    processes = [mp.Process(target=multiprocessing_main_worker, args=(inQ, outQ)) for i in range(nmbr_processes)]
    for proc in processes:
        proc.start()
    # --- now let's do our actual stuff ---
    start = time.time()

    a = pn * d.imgY/nmbr_processes
    b = (pn + 1) * d.imgY/nmbr_processes
    slow_calculataion(imgMatrix[a:b, :, :], d)
    pn += 1

    time_parallel = time.time() - start
    # --- now close everything properly ---
    for i in range(nmbr_processes):
        inQ.put(None)
    # --- now functions have stopped, now we close the process on our computer ----
    for proc in processes:
        proc.join()
    print(f"time taken using parallel computing: {time_parallel} seconds on {nmbr_processes} cores")
    return ()

def multiprocessing_main_worker(inQ, outQ):
    '''
    This function will continuously check for a new job that has been added to the input Que,
    perform the job and append the result to the output Que.

    Technically this is the only function you are sending to all Processes


    :param inQ: instance of mp.Queue() to read in parameters to send to slow process
    :param outQ: instance of mp.Queue() to store results
    :return:
    '''

    while True:
        # -- Check if there is a new job loaded to the Que (first core that picks it up will do the trick)
        job = inQ.get()
        if job is None:
            break
        # unpack parameters from job specification
        waiting_time = job[0]
        # perform calculation
        output = slow_calculataion(imgMatrix, d)
        outQ.put(output)

