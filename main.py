

from absl import app
from run_experiment import run
from proc_results import main_proc, video_for_tunning
# from rank_unlbl  import rank_unlbl_data, map_unlbl_data

def model_selection(argv):
	extend_save_idx = run() 

	# # to process learning curve
	# extend_save_idx = "002"
	main_proc(extend_save_idx)

	# # to process video for tunning
	# video_for_tunning(ith_trial=extend_save_idx)

	# # rank unlabel data 
	# rank_unlbl_data(ith_trial=extend_save_idx)

	# map_unlbl_data(ith_trial=extend_save_idx)


if __name__ == "__main__":
  app.run(model_selection) 