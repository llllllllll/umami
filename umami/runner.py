import sys
import os
import traceback
import time

from  .ExperimentGrid import ExperimentGrid
from .helpers import (
    log,
    redirect_output,
    job_output_file,
    save_job,
    load_job,
    job_file_for,
)


# System dependent modules
DEFAULT_MODULES = [ 'packages/epd/7.1-2',
                    'packages/matlab/r2011b',
                    'mpi/openmpi/1.2.8/intel',
                    'libraries/mkl/10.0',
                    'packages/cuda/4.0',
                    ]

MCR_LOCATION = "/home/matlab/v715" # hack


def job_runner(job):
    '''This fn runs in a new process.  Now we are going to do a little
    bookkeeping and then spin off the actual job that does whatever it is we're
    trying to achieve.'''

    redirect_output(job_output_file(job))
    log("Running in wrapper mode for '%s'\n" % (job['id']))

    ExperimentGrid.job_running(job['expt_dir'], job['id'])

    # Update metadata and save the job file, which will be read by the job wrappers.
    job['start_t'] = int(time.time())
    job['status']  = 'running'
    save_job(job)

    success    = False
    start_time = time.time()

    try:
        # Convert the PB object into useful parameters.
        params = {}
        params = {
            param.name: param.value
            for param in job['param']
        }

        ns = {}
        with open(os.path.join(job['expt_dir'], job['main-file'])) as f:
            exec(f.read(), ns)

        result = ns['main'](job['id'], params)

        log("Got result %f\n" % (result))

        # Store the result.
        job['value'] = result
        save_job(job)

        success = True
    except:
        log("-" * 40)
        log("Problem running the job:")
        log(sys.exc_info())
        log(traceback.print_exc(limit=1000))
        log("-" * 40)

    end_time = time.time()
    duration = end_time - start_time

    # The job output is written back to the job file, so we read it back in to
    # get the results.
    job_file = job_file_for(job)
    job      = load_job(job_file)

    log("Job file reloaded.")

    if 'value' not in job:
        log("Could not find value in output file.")
        success = False

    if success:
        log("Completed successfully in %0.2f seconds. [%f]"
                         % (duration, job['value']))

        # Update the status for this job.
        ExperimentGrid.job_complete(job['expt_dir'], job['id'],
                                    job['value'], duration)
        job['status'] = 'complete'
    else:
        log("Job failed in %0.2f seconds." % (duration))

        # Update the experiment status for this job.
        ExperimentGrid.job_broken(job['expt_dir'], job['id'])
        job['status'] = 'broken'

    job['end_t'] = int(time.time())
    job['duration'] = duration

    save_job(job)
