#include <iostream>
#include <vector>
#include <cmath>

class GradientBoosting {
public:
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int numIterations, double learningRate) {
        this->numIterations = numIterations;
        this->learningRate = learningRate;
        models.clear();
        residuals = y;

        for (int i = 0; i < numIterations; ++i) {
            double prediction = 0.0;
            for (int j = 0; j < X.size(); ++j) {
                prediction += learningRate * predict(X[j]);
            }

            std::vector<double> gradients(X.size());
            for (int j = 0; j < X.size(); ++j) {
                gradients[j] = 2 * (prediction - y[j]);
            }

            models.push_back(gradients);
            updateResiduals(gradients);
        }
    }

    double predict(const std::vector<double>& sample) {
        double prediction = 0.0;
        for (int i = 0; i < models.size(); ++i) {
            prediction += learningRate * models[i][0];
        }
        return prediction;
    }

private:
    std::vector<std::vector<double>> models;
    std::vector<double> residuals;
    int numIterations;
    double learningRate;

    void updateResiduals(const std::vector<double>& gradients) {
        for (int i = 0; i < residuals.size(); ++i) {
            residuals[i] -= learningRate * gradients[i];
        }
    }
};

int main() {
    GradientBoosting model;

    std::vector<std::vector<double>> X = {{2.0, 3.0}, {4.0, 1.0}, {5.0, 7.0}};
    std::vector<double> y = {0.0, 1.0, 0.0};

    model.fit(X, y, 100, 0.1);

    std::vector<double> sample = {3.0, 2.0};
    double prediction = model.predict(sample);

    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
