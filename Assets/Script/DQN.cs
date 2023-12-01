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
         * 만들어야할것
         * 
         * 우선 DNN은 만든거 같으니
         * replayMemory를 만들자
         * 그전에 입실론 그리디를 먼저 만들어야하나?
         * 실제액션
         * 학습
         * DNN 인풋 아웃풋 만들기
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
            //인풋설정
            Dnn_ = new DNN(InputNodes,learningRate);
            replayMemory = new ReplayMemory(10000);
            //은닉층
            foreach (int node in HiddenLayer)
            {
                Dnn_.AddLayer(node);
            }
            //아웃풋설정
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
            { //랜덤행동
                Action = Random.Range(0, OutputNode);
            }
            else
            { //그리디 행동
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
                //다음 q 값계산
                Matrix<double> nextQMatrix = Dnn_Target.Forward(replay.nextState);
                int action = GreedyAction(nextQMatrix);
                double nextqValue = getQVlaue(nextQMatrix, action);

                //현재q값으로 학습
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

        //매트릭스 크리를 넣고 해당 action에다가만 qvalue를 부여함
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
            //maxsize보다 많아지면 가장 앞에것을 제거
            if(memory.Count > maxSize) memory.RemoveAt(0);
        }
        public Replay Get(int i) => memory[i];
        public int Count() => memory.Count;
    }
    public class Replay
    {
        public Matrix<double> currentState; //현제 상태
        public int Action; //행동
        public double reward; //보상
        public Matrix<double> nextState; //다음 상태

        public Replay(Matrix<double> currentState, int Action, double reward, Matrix<double> nextState)
        {
            this.currentState = currentState;
            this.Action = Action;
            this.reward = reward;
            this.nextState = nextState;
        }
    }
}
