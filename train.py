import pandas as pd
import numpy as np
from datetime import datetime, time
import pickle
import ast

def train_model(data_files):
    all_data = pd.concat([pd.read_csv(file, index_col=0) for file in data_files])
    all_data['time'] = pd.to_datetime(all_data['time'], format='%H:%M:%S').dt.time
    
    priors = {}
    sensor_reliability = {}

    # Calculate priors
    for room in [f'r{i}' for i in range(1, 35)]:
        room_data = all_data.groupby('time')[room].agg(lambda x: (x > 0).mean())
        full_time_range = [time(hour=h, minute=m) for h in range(8, 18) for m in range(0, 60, 15)]
        priors[room] = room_data.reindex(full_time_range, fill_value=0).to_dict()

    # Calculate sensor reliability
    motion_sensors = {
        'motion_sensor1': 'r1', 'motion_sensor2': 'r14', 'motion_sensor3': 'r19',
        'motion_sensor4': 'r28', 'motion_sensor5': 'r29', 'motion_sensor6': 'r32'
    }
    cameras = {
        'camera1': 'r3', 'camera2': 'r21', 'camera3': 'r25', 'camera4': 'r34'
    }

    for sensor, room in motion_sensors.items():
        true_positives = ((all_data[sensor] == 'motion') & (all_data[room] > 0)).sum()
        false_positives = ((all_data[sensor] == 'motion') & (all_data[room] == 0)).sum()
        true_negatives = ((all_data[sensor] == 'no motion') & (all_data[room] == 0)).sum()
        false_negatives = ((all_data[sensor] == 'no motion') & (all_data[room] > 0)).sum()
        
        reliability_motion = true_positives / (true_positives + false_negatives)
        reliability_no_motion = true_negatives / (true_negatives + false_positives)
        
        sensor_reliability[sensor] = {
            'motion': reliability_motion,
            'no motion': reliability_no_motion
        }

    for sensor, room in cameras.items():
        true_positives = ((all_data[sensor] > 0) & (all_data[room] > 0)).sum()
        true_negatives = ((all_data[sensor] == 0) & (all_data[room] == 0)).sum()
        total = len(all_data)
        reliability = (true_positives + true_negatives) / total
        sensor_reliability[sensor] = reliability

    # Calculate robot reliability
    for robot in ['robot1', 'robot2']:
        correct = 0
        total = 0
        for _, row in all_data.iterrows():
            robot_data = ast.literal_eval(row[robot])
            robot_room, robot_count = robot_data
            if robot_room.startswith('r'):
                actual_count = row[robot_room]
                if (robot_count > 0 and actual_count > 0) or (robot_count == 0 and actual_count == 0):
                    correct += 1
                total += 1
        sensor_reliability[robot] = correct / total if total > 0 else 0

    # Learn DBN parameters
    transition_probs = {room: {'stay_occupied': 0, 'become_occupied': 0} for room in priors.keys()}
    total_transitions = {room: {'from_occupied': 0, 'from_unoccupied': 0} for room in priors.keys()}

    for _, data in all_data.groupby(all_data.index // 4):  # Group by 1-minute intervals
        for room in priors.keys():
            prev_state = data[room].iloc[0] > 0
            for state in data[room].iloc[1:] > 0:
                if prev_state:
                    total_transitions[room]['from_occupied'] += 1
                    if state:
                        transition_probs[room]['stay_occupied'] += 1
                else:
                    total_transitions[room]['from_unoccupied'] += 1
                    if state:
                        transition_probs[room]['become_occupied'] += 1
                prev_state = state

    for room in transition_probs:
        transition_probs[room]['stay_occupied'] /= max(total_transitions[room]['from_occupied'], 1)
        transition_probs[room]['become_occupied'] /= max(total_transitions[room]['from_unoccupied'], 1)


    return priors, sensor_reliability, transition_probs

if __name__ == "__main__":
    priors, sensor_reliability, transition_probs = train_model(['data1.csv', 'data2.csv'])
    
    with open('priors.pickle', 'wb') as f:
        pickle.dump(priors, f)
    with open('sensor_reliability.pickle', 'wb') as f:
        pickle.dump(sensor_reliability, f)
    with open('transition_probs.pickle', 'wb') as f:
        pickle.dump(transition_probs, f)
    
    print("Training completed. Priors and sensor reliabilities saved.")

    # Print all results
    print("\nPrior probabilities for all rooms and time slots:")
    for room in [f'r{i}' for i in range(1, 35)]:
        print(f"\nRoom {room}:")
        for t, prob in priors[room].items():
            print(f"  {t.strftime('%H:%M')}: {prob:.4f}")
    
    print("\nSensor Reliabilities:")
    for sensor, reliability in sensor_reliability.items():
        if isinstance(reliability, dict):
            print(f"{sensor}:")
            print(f"  motion: {reliability['motion']:.4f}")
            print(f"  no motion: {reliability['no motion']:.4f}")
        else:
            print(f"{sensor}: {reliability:.4f}")
    
    print("\nTransition Probabilities:")
    for room, probs in transition_probs.items():
        print(f"{room}:")
        print(f"  Stay occupied: {probs['stay_occupied']:.4f}")
        print(f"  Become occupied: {probs['become_occupied']:.4f}")