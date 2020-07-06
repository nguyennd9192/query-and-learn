

from absl import app
from run_experiment import run
from proc_results import main_proc, video_for_tunning

def main(argv):
	extend_save_idx = run() 
	# # to process learning curve
	# main_proc(extend_save_idx)

	# # to process video for tunning
	video_for_tunning(ith_trial=extend_save_idx)

if __name__ == "__main__":
  app.run(main)