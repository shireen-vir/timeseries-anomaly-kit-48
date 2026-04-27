class TimeSeriesAnomalyKit48:
    """
    A data science tool for anomaly detection in time series data.

    Attributes:
        data (list): The input time series data.
    """

    def __init__(self, data):
        self.data = data

    def detect_anomalies(self):
        # Basic implementation of anomaly detection using z-score method
        mean = sum(self.data) / len(self.data)
        std_dev = (sum((x - mean) ** 2 for x in self.data) / len(self.data)) ** 0.5
        return [x for x in self.data if abs((x - mean) / std_dev) > 2]


def main():
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate sample time series data
    np.random.seed(0)
    data = np.random.normal(size=100)
    data[50] = 10  # introduce an anomaly

    kit = TimeSeriesAnomalyKit48(data)
    anomalies = kit.detect_anomalies()

    print("Anomalies detected:", anomalies)

    # Plot the data
    plt.plot(data)
    plt.plot([i for i, x in enumerate(data) if x in anomalies], anomalies, 'ro')
    plt.show()


if __name__ == "__main__":
    main()