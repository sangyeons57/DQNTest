using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

namespace ANN
{
    public class DQN
    {
        /**
         * �������Ұ�
         * 
         * �켱 DNN�� ����� ������
         * replayMemory�� ������
         * ������ �ԽǷ� �׸��� ���� �������ϳ�?
         * �����׼�
         * �н�
         * DNN ��ǲ �ƿ�ǲ �����
        */
        private DNN Dnn_;
        private DNN Dnn_Target;
        private ReplayMemory replayMemory;
        public int InputNode { get; private set; }
        public int OutputNode { get; private set; }

        public float epsilon = 1.0f;
        private float decayingEpsilon = 0.0001f;
        public float attenuation = 0.5f;

        private int framSkipping;
        private int skip;

        private Matrix<double> lastState;

        private int lastAction = -1;

        public DQN( int InputNodes, List<int> HiddenLayer, int OutputNodes, int frameSkipping = 4 ,double learningRate = 0.005)
        {
            InputNode = InputNodes;
            OutputNode = OutputNodes;
            this.framSkipping = frameSkipping;
            //��ǲ����
            Dnn_ = new DNN(InputNodes,learningRate);
            replayMemory = new ReplayMemory(10000);
            //������
            foreach (int node in HiddenLayer)
            {
                Dnn_.AddLayer(node);
            }
            //�ƿ�ǲ����
            Dnn_.AddLayer(OutputNodes, true);
            Dnn_.DNNInit();
            Dnn_Target = Dnn_.DeepCopy();

        }

        public int EpsilonGreedyAction(Matrix<double>inputs)
        {
            skip++;
            lastState = inputs;
            Matrix<double> forward = Dnn_.Forward(inputs);
            if(skip <= framSkipping && lastAction > 0) { return lastAction; }

            int Action = -1;
            if (Random.value < this.epsilon)
            { //�����ൿ
                Action = Random.Range(0, OutputNode);
            }
            else
            { //�׸��� �ൿ
                Action = GreedyAction(forward);
            }
            skip = 0;
            lastAction = Action;
            return Action;
        }
        public void SaveReplayMemory(double reward, Matrix<double> nextState)
        {
            if (lastAction < 0) return;
            Debug.Log("rewared: " +reward);
            replayMemory.Add(new Replay(lastState, lastAction, reward, nextState));
        }

        public void Synchronize() => Dnn_Target = Dnn_.DeepCopy();
        public void DecayingEpsilon() => epsilon = Math.Max(0.1f , epsilon * (1 - decayingEpsilon));

        public void Learning(int batchSize)
        {
            Dnn_.resetDelta();
            for (int i = 0; i < batchSize; i++)
            {
                Replay replay = replayMemory.Get(Random.Range(0, replayMemory.Count()));
                //���� q �����
                Matrix<double> nextQMatrix = Dnn_Target.Forward(replay.nextState);
                int action = GreedyAction(nextQMatrix);
                double nextqValue = getQVlaue(nextQMatrix, action);

                //����q������ �н�
                Matrix<double> currentQValue = Dnn_.Forward(replay.currentState);
                Dnn_.Backword( replay.reward + nextqValue * attenuation, replay.Action);
            }
            Dnn_.updateDNN();
        }

        private int GreedyAction(Matrix<double> qValues)
        {
            int greedyAction = -1;
            double maxQvalue = double.MinValue;
            int index = 0;  
            foreach (double qValue in qValues.ToArray())
            {
                if(qValue > maxQvalue)
                {
                    maxQvalue = qValue;
                    greedyAction = index;
                }
                index++;
            }
            return greedyAction;
        }

        private double getQVlaue(Matrix<double> output, int action = -1)
        {
            for (int i = 0; i < output.ColumnCount; i++)
            {
                if (action == i) return output[0,i];
            }
            return -1;
        }

        //��Ʈ���� ũ���� �ְ� �ش� action���ٰ��� qvalue�� �ο���
        private Matrix<double> getQvalueMatrix(Matrix<double> Matrix, double qValue, int action)
        {
            if (action < 0) return null;

            for (int i = 0; i < Matrix.RowCount; i++)
            {
                if (action == i)
                    Matrix[0, i] = qValue;
                else
                    Matrix[0, i] = 0;
            }
            return Matrix;
        }
    }

    public class ReplayMemory
    {
        private List<Replay> memory;
        int maxSize;
        public ReplayMemory(int size)
        {
            maxSize = size;
            memory = new List<Replay>();
        }

        public void Add(Replay m)
        {
            memory.Add(m);
            //maxsize���� �������� ���� �տ����� ����
            if(memory.Count > maxSize) memory.RemoveAt(0);
        }
        public Replay Get(int i) => memory[i];
        public int Count() => memory.Count;
    }
    public class Replay
    {
        public Matrix<double> currentState; //���� ����
        public int Action; //�ൿ
        public double reward; //����
        public Matrix<double> nextState; //���� ����

        public Replay(Matrix<double> currentState, int Action, double reward, Matrix<double> nextState)
        {
            this.currentState = currentState;
            this.Action = Action;
            this.reward = reward;
            this.nextState = nextState;
        }
    }
}
