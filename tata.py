import logging
import random

import numpy as np
from matplotlib import pyplot as plt


_logger = logging.getLogger(__name__)

class Model:
    def __init__(self, time_model: list | tuple, expected_sensor_inputs: list | tuple, learning_rate: float, sensor_sensitivity: float):
        if len(time_model) != len(expected_sensor_inputs):
            raise ValueError("The time model and the expected sensor inputs must have the same length")

        self._learning_rate = learning_rate
        self._sensor_sensitivity = sensor_sensitivity

        self._expected_sensor_inputs = expected_sensor_inputs
        self._time_model = time_model
        self._next_time_model: list | tuple = self._time_model.copy()

        self._phase_index = 0
        self._current_time = 0.0
        self._phase_count = len(self._next_time_model)

    def predict(self, sensor_input: float, time_since_last_prediction: float) -> tuple[float, bool]:
        # Update the time
        self._current_time += time_since_last_prediction
        
        # Predict the percentage of the current phase we are in. If we went too far, stale the prediction
        capped_value = min(self._current_time, self._time_model[self._phase_index])
        value = (capped_value + sum(self._time_model[:self._phase_index])) / sum(self._time_model)
        
        # Check if we should update the model based on the sensor input
        has_updated = self._update_model(sensor_input)

        return value, has_updated

    def _update_model(self, sensor_input: float) -> bool:
        # Test if the sensor input is close to the expected value, if so, we already are in the right phase
        if abs(self._expected_sensor_inputs[self._phase_index] - sensor_input) < self._sensor_sensitivity:
            return False

        # Adjust the next model based on the prediction error it made
        prediction_error = self._current_time - self._time_model[self._phase_index]
        self._next_time_model[self._phase_index] += prediction_error * self._learning_rate

        # Advance the phase
        self._current_time = 0
        self._phase_index = (self._phase_index + 1) % self._phase_count
        
        # If we looped, transfer the next model to the current model
        if self._phase_index == 0:
            self._time_model = self._next_time_model.copy()
        
        return True
            

def main():
    logging.basicConfig(level=logging.INFO)
    
    time_window = 3
    dt = 0.01
    model = Model(time_model=[0.1, 0.1], expected_sensor_inputs=[1, 0], learning_rate=0.2, sensor_sensitivity=0.1)
    
    # Generate some data to simulate a real data acquisition
    max_step_count = 1000  # Number of steps to simulate
    target_step_time = 0.9  # Time to perform one step in seconds
    target_model = [target_step_time * 0.4, target_step_time * 0.6]  # Time for the ground phase and the swing phase
    target_model_noises = [0.05, 0.1]
    simulated_data = []
    target_percentage = []
    target_signal_frames = []
    for _ in range(max_step_count):
        phases = [random.uniform(target * (1 - noise), target * (1 + noise)) for target, noise in zip(target_model, target_model_noises)]
        frame_counts = [int(phase / dt) for phase in phases]
        step_frames = sum(([data] * count for data, count in zip(model._expected_sensor_inputs, frame_counts)), [])  # Sum flattens the list
        simulated_data += step_frames
        target_percentage += [frame / len(step_frames) for frame in range(len(step_frames))]
        target_signal_frames += frame_counts

    # Prepare the graphs
    frame_window = int(time_window / dt)
    plt.set_loglevel (level = 'warning')
    
    # Top axis is the data
    ax_signal_plot = plt.subplot(2, 1, 1)
    signal_plot = ax_signal_plot.plot(np.nan, np.nan)
    plt.ylim(-0.1, 1.1)
    
    # Bottom axis is the prediction
    ax_prediction_plot = plt.subplot(2, 1, 2)
    real_data_plot = ax_prediction_plot.plot(np.nan, np.nan)
    plt.ylim(-0.1, 1.1)
    predicted_data_plot = ax_prediction_plot.plot(np.nan, np.nan)
    plt.ylim(-0.1, 1.1)
    # ax_prediction_plot.legend(["Real", "Predicted"])

    plt.ion()
    plt.draw()
    plt.show()

    # Run the simulation
    t = []
    predictions = []
    changed_phase_times = []
    for i in range (len(simulated_data)):
        sensor_input = simulated_data[i]

        # Keep some information for logging (that must be done before the prediction)
        real_update_time = model._current_time
        predicted_update_time = model._time_model[model._phase_index]
        
        # Predict the percentage of the current phase we are in
        value, model_has_updated = model.predict(sensor_input=sensor_input, time_since_last_prediction=dt)
        
        # Store the values
        t.append(i * dt)
        predictions.append(value)
        if model_has_updated:
           changed_phase_times.append(i * dt)
        
        _logger.debug(f"Prediction: {value * 100:.2f} (real: {target_percentage[i] * 100:.2f}%)")
        if model_has_updated:
            _logger.debug(f"Change of phase to {model._phase_index} at {predicted_update_time} (real: {real_update_time})")
        
        # Update the graphs so it show the last time_window seconds
        simulated_frames_min = i - frame_window if i > frame_window else 0
        signal_plot[0].set_data(t[simulated_frames_min:], simulated_data[simulated_frames_min:i+1])
        real_data_plot[0].set_data(t[simulated_frames_min:], target_percentage[simulated_frames_min:i+1])
        predicted_data_plot[0].set_data(t[simulated_frames_min:], predictions[-frame_window-1:])
        
        ax_signal_plot.set_xlim(t[simulated_frames_min], t[-1])
        ax_prediction_plot.set_xlim(t[simulated_frames_min], t[-1])
        # plt.draw()
        plt.pause(0.001)
        
    _logger.info(f"Final model: {model._time_model} (real: {target_model})")




if __name__ == "__main__":
    main()