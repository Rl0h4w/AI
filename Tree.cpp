#include <iostream>
#include <vector>
#include <cmath>

struct Node {
    int featureIndex;
    double threshold;
    int leftChildIndex;
    int rightChildIndex;
    int predictedClass;
};

class DecisionTree {
public:
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int maxDepth) {
        this->maxDepth = maxDepth;
        tree.clear();
        buildTree(X, y, 0);
    }

    int predict(const std::vector<double>& sample) {
        int currentNodeIndex = 0;
        while (tree[currentNodeIndex].leftChildIndex != -1 && tree[currentNodeIndex].rightChildIndex != -1) {
            if (sample[tree[currentNodeIndex].featureIndex] <= tree[currentNodeIndex].threshold) {
                currentNodeIndex = tree[currentNodeIndex].leftChildIndex;
            } else {
                currentNodeIndex = tree[currentNodeIndex].rightChildIndex;
            }
        }
        return tree[currentNodeIndex].predictedClass;
    }

private:
    std::vector<Node> tree;
    int maxDepth;

    void buildTree(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int currentNodeIndex, int depth = 0) {
        if (depth >= maxDepth || y.size() == 0) {
            tree.push_back({-1, -1, -1, -1, getMostCommonClass(y)});
            return;
        }

        int bestFeatureIndex = -1;
        double bestThreshold = 0.0;
        double bestImpurity = std::numeric_limits<double>::max();
        std::vector<int> bestLeftIndices;
        std::vector<int> bestRightIndices;

        for (int featureIndex = 0; featureIndex < X[0].size(); ++featureIndex) {
            for (int i = 0; i < X.size(); ++i) {
                std::vector<int> leftIndices;
                std::vector<int> rightIndices;

                for (int j = 0; j < X.size(); ++j) {
                    if (X[j][featureIndex] <= X[i][featureIndex]) {
                        leftIndices.push_back(j);
                    } else {
                        rightIndices.push_back(j);
                    }
                }

                double impurity = calculateImpurity(y, leftIndices, rightIndices);
                if (impurity < bestImpurity) {
                    bestImpurity = impurity;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = X[i][featureIndex];
                    bestLeftIndices = leftIndices;
                    bestRightIndices = rightIndices;
                }
            }
        }

        if (bestFeatureIndex == -1) {
            tree.push_back({-1, -1, -1, -1, getMostCommonClass(y)});
            return;
        }

        tree.push_back({bestFeatureIndex, bestThreshold, -1, -1, -1});
        int leftChildIndex = tree.size() - 1;
        buildTree(X, y, leftChildIndex, depth + 1);

        tree[currentNodeIndex].leftChildIndex = leftChildIndex;

        tree.push_back({-1, -1, -1, -1, -1});
        int rightChildIndex = tree.size() - 1;
        buildTree(X, y, rightChildIndex, depth + 1);

        tree[currentNodeIndex].rightChildIndex = rightChildIndex;
    }

    int getMostCommonClass(const std::vector<int>& y) {
        std::vector<int> classCounts(10, 0);
        for (int i = 0; i < y.size(); ++i) {
            classCounts[y[i]]++;
        }
        int mostCommonClass = 0;
        int maxCount = 0;
        for (int i = 0; i < classCounts.size(); ++i) {
            if (classCounts[i] > maxCount) {
                maxCount = classCounts[i];
                mostCommonClass = i;
            }
        }
        return mostCommonClass;
    }

    double calculateImpurity(const std::vector<int>& y, const std::vector<int>& leftIndices, const std::vector<int>& rightIndices) {
        double impurity = 0.0;
        if (leftIndices.size() > 0) {
            std::vector<int> leftClassCounts(10, 0);
            for (int i = 0; i < leftIndices.size(); ++i) {
                leftClassCounts[y[leftIndices[i]]]++;
            }
            for (int i = 0; i < leftClassCounts.size(); ++i) {
                double p = static_cast<double>(leftClassCounts[i]) / leftIndices.size();
                impurity -= p * std::log2(p + 1e-10);
            }
        }
        if (rightIndices.size() > 0) {
            std::vector<int> rightClassCounts(10, 0);
            for (int i = 0; i < rightIndices.size(); ++i) {
                rightClassCounts[y[rightIndices[i]]]++;
            }
            for (int i = 0; i < rightClassCounts.size(); ++i) {
                double p = static_cast<double>(rightClassCounts[i]) / rightIndices.size();
                impurity -= p * std::log2(p + 1e-10);
            }
        }
        return impurity;
    }
};

int main() {
    DecisionTree model;

    std::vector<std::vector<double>> X = {{2.0, 3.0}, {4.0, 1.0}, {5.0, 7.0}};
    std::vector<int> y = {0, 1, 0};

    model.fit(X, y, 2);

    std::vector<double> sample = {3.0, 2.0};
    int predictedClass = model.predict(sample);

    std::cout << "Predicted class: " << predictedClass << std::endl;

    return 0;
}
