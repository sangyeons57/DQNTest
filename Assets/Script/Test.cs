using ANN;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using MathNet.Numerics.Statistics;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;

public class Test : MonoBehaviour
{
    DQN DQN_;
    public double notRecievedReward;
    public GameObject target;

    private int syncStep;
    void Start()
    {
        //인풋 6: 내x z, 보상 x z , 거리,내 rotate y
        //액션 4: 왼쪽회전, 오른쪽 회전, 직진, 후진 
        List<int>HiddenLayer = new List<int>() { 7,6,5 };
        DQN_ = new DQN(6,HiddenLayer,3);
    }

    float distane1;
    private void Update()
    {
        notRecievedReward = 0;
        Matrix<double> inputMatrix = Matrix<double>.Build.DenseOfArray(new double[,]
        { { transform.position.x, transform.position.z, target.transform.position.x, target.transform.position.z
        , Vector3.Distance(transform.position, target.transform.position) , transform.rotation.y} });
        
        int Action = DQN_.EpsilonGreedyAction(inputMatrix);

        distane1 = Vector3.Distance(transform.position, target.transform.position);
        if(Action == 0)
            transform.GetComponent<Rigidbody>().angularVelocity += new Vector3(0,0.7f,0);
        else if(Action == 1)
            transform.GetComponent<Rigidbody>().angularVelocity += new Vector3(0,-0.7f,0);
        else if(Action == 2)
            gameObject.GetComponent<Rigidbody>().velocity += transform.TransformDirection(Vector3.forward) * 1f;

        if (transform.position.y < 0)
        {
            notRecievedReward -= 2;
            transform.position = new Vector3(0, 1, 0);
            transform.rotation = Quaternion.identity;
            gameObject.GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 0);
            gameObject.GetComponent<Rigidbody>().angularVelocity = new Vector3(0, 0, 0);
        }
        if (distane1 > Vector3.Distance(transform.position, target.transform.position))
            notRecievedReward += 0.0001f;
        else
            notRecievedReward -= 0.0001f;

        //새로온 상태 매트릭스
        Matrix<double> newStateMatrix = Matrix<double>.Build.DenseOfArray(new double[,]
        { { transform.position.x, transform.position.z, target.transform.position.x, target.transform.position.z
        , Vector3.Distance(transform.position, target.transform.position) , transform.rotation.y} });
        
        Debug.Log("R=: " + notRecievedReward);
        DQN_.SaveReplayMemory(notRecievedReward,newStateMatrix);
        DQN_.DecayingEpsilon();
        DQN_.Learning(16);

        syncStep++;
        if(syncStep == 5)
        {
            DQN_.Synchronize();
            syncStep = 0;
        }
    }


    class Program
{
    public void start()
    {
        var inputs = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
        var targets = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0 }, { 0, 0.5 }, { 0, 0.5 }, { 1, 1 } });

        var weights1 = Matrix<double>.Build.Random(2, 4);
        var weights2 = Matrix<double>.Build.Random(4, 2);

        var bias1 = Matrix<double>.Build.Dense(1, 4);
        var bias2 = Matrix<double>.Build.Dense(1, 2);

        double learningRate = 0.1;
        int epochs = 5000;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;

            for (int i = 0; i < inputs.RowCount; i++)
            {
                var inputLayer = inputs.Row(i).ToRowMatrix();
                var target = targets.Row(i).ToRowMatrix();

                var layer1 = ReLU(inputLayer * weights1 + bias1);
                var layer2 = ReLU(layer1 * weights2 + bias2);

                var layer2Error = target - layer2;
                totalError += Math.Pow(layer2Error.L2Norm(), 2);

                var layer2Delta = layer2Error.PointwiseMultiply(ReLUDerivative(layer2));
                var layer1Error = layer2Delta * weights2.Transpose();
                var layer1Delta = layer1Error.PointwiseMultiply(ReLUDerivative(layer1));

                weights2 += layer1.Transpose() * layer2Delta * learningRate;
                weights1 += inputLayer.Transpose() * layer1Delta * learningRate;

                bias2 += layer2Delta * learningRate;
                bias1 += layer1Delta * learningRate;
            }

            if (epoch % 1000 == 0)
            {
                Debug.Log($"Epoch: {epoch}, Error: {totalError}");
            }
        }
    }


    static Matrix<double> ReLU(Matrix<double> matrix)
    {
        return matrix.Map(x => x > 0 ? x : 0.01 * x);
    }

    static Matrix<double> ReLUDerivative(Matrix<double> matrix)
    {
        return matrix.Map(x => x > 0 ? 1.0 : 0.01);
    }

}

class Program2
{
    static Matrix<double> Relu(Matrix<double> x)
    {
        return x.Map(v => Math.Max(0, v));
    }

    static Matrix<double> ReluDerivative(Matrix<double> x)
    {
        return x.PointwiseMultiply(x.Map(v => v > 0 ? 1.0 : 0.0));
    }

    public double start()
    {
        // 입력 데이터
        var inputs = new List<Matrix<double>>();
            inputs.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0 } }));
            inputs.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 } }));
            inputs.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0 } }));
            inputs.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1 } }));

        // 타겟 데이터
        var targets = new List<Matrix<double>>();
            targets.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0 } }));
            targets.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0.5 } }));
            targets.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0.5 } }));
            targets.Add(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1 } }));



                    // 가중치 초기화
        var weights1 = Matrix<double>.Build.Random(2, 4) * Math.Sqrt(2.0/4);  // Update: Change from 2 to 4
        var weights2 = Matrix<double>.Build.Random(4, 5) * Math.Sqrt(2.0/5);  // Update: Change from 2 to 4
        var weights3 = Matrix<double>.Build.Random(5, 6) * Math.Sqrt(2.0/6);  // New: Add weights for the third hidden layer
        var weights4 = Matrix<double>.Build.Random(6, 2) * Math.Sqrt(2.0 / 2);  // New: Add weights for the output layer

        // 편향 초기화
        var bias1 = Matrix<double>.Build.Dense(1, 4);  // Update: Change from 2 to 4
        var bias2 = Matrix<double>.Build.Dense(1, 5);  // Update: Change from 2 to 4
        var bias3 = Matrix<double>.Build.Dense(1, 6);  // New: Add bias for the third hidden layer
        var bias4 = Matrix<double>.Build.Dense(1, 2);  // New: Add bias for the output layer


        // 학습률
        var learningRate = 0.01;

        // 에포크 수
        var epochs = 10000;

        double totalError = 0;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            totalError = 0;

            for (int i = 0; i < inputs.Count; i++)
            {
                // 전방향 전파
                //var layer1 = Relu(inputs.Row(i).ToColumnMatrix() * weights1 + bias1);


                var layer1 = Relu(inputs[i] * weights1 + bias1);
                var layer2 = Relu(layer1 * weights2 + bias2);
                var layer3 = Relu(layer2 * weights3 + bias3);  // New: Add forward propagation for the third hidden layer
                var layer4 = Relu(layer3 * weights4 + bias4); 

                // 오차 계산
                var outputError = targets[i] - layer4;  // New: Use layer4 for error calculation
                    totalError += outputError.PointwisePower(2).RowSums().Mean();

                // 역전파
                var layer4Delta = outputError.PointwiseMultiply(ReluDerivative(layer4));
                var layer3Error = layer4Delta * weights4.Transpose();
                var layer3Delta = layer3Error.PointwiseMultiply(ReluDerivative(layer3));
                var layer2Error = layer3Delta * weights3.Transpose();
                var layer2Delta = layer2Error.PointwiseMultiply(ReluDerivative(layer2));
                var layer1Error = layer2Delta * weights2.Transpose();
                var layer1Delta = layer1Error.PointwiseMultiply(ReluDerivative(layer1));

                // 가중치 업데이트
                weights4 += layer3.Transpose() * layer4Delta * learningRate;  // New: Update weights for the output layer
                weights3 += layer2.Transpose() * layer3Delta * learningRate;  // New: Update weights for the third hidden layer
                weights2 += layer1.Transpose() * layer2Delta * learningRate;
                weights1 += inputs[i].Transpose() * layer1Delta * learningRate;

                // 편향 업데이트
                bias4 += layer4Delta * learningRate;  // New: Update bias for the output layer
                bias3 += layer3Delta * learningRate;  // New: Update bias for the third hidden layer
                bias2 += layer2Delta * learningRate;
                bias1 += layer1Delta * learningRate;
            }

            if (epoch % 1000 == 0)
            {
                Debug.Log($"Epoch: {epoch}, Error: {totalError}");
            }
        }
        return totalError;
    }
}


}
