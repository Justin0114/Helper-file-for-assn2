from datetime import datetime, time
from Bayes_solution import get_action, DynamicBayesianNetwork, calculate_posterior, priors, motion_sensors, cameras, transition_probs
from example_test import SmartBuildingSimulatorExample
import itertools

def evaluate_performance(params):
    # Unpack parameters
    dbn_prior_weight, final_prior_weight, final_predicted_weight, final_posterior_weight, threshold = params
    
    # Update the DynamicBayesianNetwork class
    class ModifiedDBN(DynamicBayesianNetwork):
        def predict(self, room, time_slot):
            prior = priors[room].get(time_slot, 0)
            prev_state = self.states[room]
            stay_occupied = self.transition_probs[room]['stay_occupied']
            become_occupied = self.transition_probs[room]['become_occupied']
            
            predicted_prob = prev_state * stay_occupied + (1 - prev_state) * become_occupied
            return (1 - dbn_prior_weight) * predicted_prob + dbn_prior_weight * prior

    # Create an instance of the modified DBN
    dbn = ModifiedDBN([f'r{i}' for i in range(1, 35)], transition_probs)

    # Update the get_action function
    def modified_get_action(sensor_data):
        actions_dict = {}
        current_time = sensor_data['time']
        current_15min = time(hour=current_time.hour, minute=(current_time.minute // 15) * 15)

        if current_time >= time(18, 0):
            current_15min = time(17, 45)

        for room in [f'r{i}' for i in range(1, 35)]:
            light_key = f'lights{room[1:]}'
            
            prior_prob = priors[room].get(current_15min, 0)
            predicted_prob = dbn.predict(room, current_15min)
            
            has_relevant_sensor = (room in motion_sensors.values() or 
                                   room in cameras.values() or 
                                   any((sensor_data.get(robot) is not None and 
                                        isinstance(sensor_data[robot], tuple) and 
                                        len(sensor_data[robot]) == 2 and
                                        sensor_data[robot][0] == room) 
                                       for robot in ['robot1', 'robot2']))

            if has_relevant_sensor:
                posterior_prob = calculate_posterior(predicted_prob, sensor_data, room)
                dbn.update(room, posterior_prob)
                final_prob = final_prior_weight * prior_prob + final_predicted_weight * predicted_prob + final_posterior_weight * posterior_prob
            else:
                final_prob = (final_prior_weight / (final_prior_weight + final_predicted_weight)) * prior_prob + (final_predicted_weight / (final_prior_weight + final_predicted_weight)) * predicted_prob

            if final_prob >= threshold:
                actions_dict[light_key] = 'on'
            else:
                actions_dict[light_key] = 'off'

        return actions_dict

    # Run simulation
    simulator = SmartBuildingSimulatorExample()
    total_cost = 0
    for _ in range(len(simulator.data)):
        sensor_data = simulator.timestep()
        actions_dict = modified_get_action(sensor_data)   
        total_cost += simulator.cost_timestep(actions_dict)

    return total_cost

# The rest of your code (grid_search function and main block) remains the same

import itertools
from tqdm import tqdm

def grid_search():
    # Define parameter ranges
    dbn_prior_weights = [0.5, 0.6, 0.7, 0.8, 0.9]
    final_prior_weights = [0.2, 0.3, 0.4, 0.5, 0.6]
    final_predicted_weights = [0.2, 0.3, 0.4, 0.5, 0.6]
    final_posterior_weights = [0.2, 0.3, 0.4, 0.5, 0.6]
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]

    best_params = None
    best_cost = float('inf')

    total_combinations = len(dbn_prior_weights) * len(final_prior_weights) * len(final_predicted_weights) * len(final_posterior_weights) * len(thresholds)
    print(f"Total combinations to test: {total_combinations}")

    # Create a progress bar
    pbar = tqdm(total=total_combinations, desc="Parameter Search Progress")

    for params in itertools.product(dbn_prior_weights, final_prior_weights, final_predicted_weights, final_posterior_weights, thresholds):
        # Ensure weights sum to 1
        if abs(params[1] + params[2] + params[3] - 1) > 1e-6:
            pbar.update(1)
            continue

        cost = evaluate_performance(params)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params

        pbar.update(1)

    pbar.close()
    return best_params, best_cost

if __name__ == "__main__":
    best_params, best_cost = grid_search()
    print(f"\nBest parameters: {best_params}")
    print(f"Best cost: {best_cost} cents")