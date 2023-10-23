# This is a sample Python script.


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import time
import multiprocessing as mp

def compute_series(nmbr_processes):
   start = time.time()
   for n in range(nmbr_processes):
       slow_calculation(waiting_time=5.0)
   time_series = time.time() - start
   return f"time taken using for-loop: {time_series} seconds"

def compute_parallel(nmbr_processes):
   # ---- first initialise it, to not have this count to total time ---
   # --- initialize the multiprocessing ----
   inQ = mp.Queue()
   outQ = mp.Queue()
   processes = [mp.Process(target=multiprocessing_main_worker, args=(inQ, outQ)) for i in range(nmbr_processes)]
   for proc in processes:
       proc.start()

   # --- now let's do our actual stuff ---
   start = time.time()
   slow_calculation(waiting_time=5.0)
   time_parallel = time.time() - start

   # --- now close everything properly ---
   # --- use our own 'stop condition in the worker', by saying the next job is 'None'
   for i in range(nmbr_processes):
       inQ.put(None)

   # --- now functions have stopped, now we close the process on our computer ----
   for proc in processes:
       proc.join()

   return f"time taken using parallel computing: {time_parallel} seconds on {nmbr_processes} cores"

def slow_calculation(waiting_time):
   '''
   silly example that just waits for 5 seconds
   :return:
   '''
   time.sleep(waiting_time)
   return f"I waited for {waiting_time} seconds"

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
       output = slow_calculation(waiting_time)
       outQ.put(output)

if __name__ == '__main__':

   nmbr_processes = 3
   # ---- compute in series first -----
   print(compute_series(nmbr_processes))

   # ---- now compute in parallel ----
   print(compute_parallel(nmbr_processes))

