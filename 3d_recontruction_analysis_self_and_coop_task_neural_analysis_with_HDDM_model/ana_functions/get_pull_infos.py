import numpy as np
import pandas as pd

def get_pull_infos(animal1, animal2, time_point_pull1, time_point_pull2, time_point_juice1, time_point_juice2):
    pull_infos = {}

    def compute_infos(pull_times, juice_times, animal_name):
        # Compute successful pulls: the last pull before each juice
        successful_pulls = [pull_times[pull_times < juice].max() for juice in juice_times if (pull_times < juice).any()]
        successful_pulls = pd.Series(successful_pulls).drop_duplicates().sort_values().reset_index(drop=True)

        failed_pulls = pull_times[~pull_times.isin(successful_pulls)]

        # Initialize lists
        num_preceding_failpull = []
        time_from_last_reward = []
        last_successful_pull_time = -np.inf

        # Create lookup-friendly series
        pull_times_sorted = pull_times.sort_values().reset_index(drop=True)

        for curr_pull in pull_times_sorted:
            # Update last successful pull if current one is successful
            if curr_pull in successful_pulls.values:
                last_successful_pull_time = curr_pull

            # Count number of failed pulls between last successful pull and current pull
            if np.isfinite(last_successful_pull_time):
                failed_between = failed_pulls[(failed_pulls > last_successful_pull_time) & (failed_pulls < curr_pull)]
            else:
                failed_between = failed_pulls[failed_pulls < curr_pull]

            num_preceding_failpull.append(len(failed_between))

            # Time from most recent juice
            past_juices = juice_times[juice_times < curr_pull]
            if len(past_juices) == 0:
                time_from_last_reward.append(np.nan)
            else:
                time_from_last_reward.append(curr_pull - past_juices.max())

        # Save the data
        pull_infos[(animal_name, 'num_preceding_failpull')] = pd.Series(num_preceding_failpull, index=pull_times_sorted.index)
        pull_infos[(animal_name, 'time_from_last_reward')] = pd.Series(time_from_last_reward, index=pull_times_sorted.index)

    # Process both animals
    compute_infos(time_point_pull1, time_point_juice1, animal1)
    compute_infos(time_point_pull2, time_point_juice2, animal2)

    return pull_infos

