using NumSharp;
using System;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;

namespace CrashCourse
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Constants
            var x = tf.constant(2);
            var y = tf.constant(10);
            var hi = tf.constant("hi");

            Console.WriteLine(x);
            Console.WriteLine(y);
            Console.WriteLine(hi);

            print(hi);

            print((int)x.numpy());
            print((int)y.numpy());
            print((string)hi.numpy());

            // Basic operations
            var add = tf.add(x, y);
            var multi = tf.multiply(x, y);

            print((int)add.numpy());
            print((int)multi.numpy());

            var mean = tf.reduce_mean(new[] { x, y });
            var sum = tf.reduce_sum(new[] { x, y });

            print((int)mean.numpy());
            print((int)sum.numpy());

            // MNIST example

            // Eager execution is turned on by default.
            // Turn it off to be able to create placeholders.
            // Eager execution evaluates operations immediatesly without building graphs.
            tf.compat.v1.disable_eager_execution();

            var mnistData = await MnistModelLoader.LoadAsync("./train", oneHot: true, trainSize: null, validationSize: 5000, testSize: null, showProgressInConsole: true);

            var mnistShape = tf.placeholder(tf.float32, new TensorShape(-1, 28 * 28));
            var mnistOutput = tf.placeholder(tf.float32, new TensorShape(-1, 10));

            var W = tf.Variable(tf.zeros(new Shape(784, 10)));
            var b = tf.Variable(tf.zeros(new Shape(10)));

            var model = tf.nn.softmax(tf.matmul(mnistShape, W) + b);

            var crossEntropy = tf.reduce_mean(-tf.reduce_sum(mnistOutput * tf.log(model), reduction_indices: 1));

            var optimizer = tf.train.GradientDescentOptimizer(0.01f).minimize(crossEntropy);

            var totalBatches = mnistData.Train.NumOfExamples / 100;

            using var session = tf.Session();

            session.run(tf.global_variables_initializer());

            var accuracy = 0f;

            // Train
            foreach (var epoch in range(20))
            {
                var avg_cost = 0.0f;

                foreach (var i in range(totalBatches))
                {
                    var start = i * 100;
                    var end = (i + 1) * 100;
                    var (batch_xs, batch_ys) = mnistData.GetNextBatch(mnistData.Train.Data, mnistData.Train.Labels, start, end);
                    // Run optimization op (backprop) and cost op (to get loss value)
                    (_, float c) = session.run((optimizer, crossEntropy),
                        (mnistShape, batch_xs),
                        (mnistOutput, batch_ys));

                    // Compute average loss
                    avg_cost += c / totalBatches;
                }

                // Display logs per epoch step
                var newEpoc = epoch + 1;

                if (newEpoc % 1 == 0)
                    print($"Epoch: {newEpoc} Cost: {avg_cost:G9}");
            }

            print("Optimization Finished!");
            // SaveModel(sess);

            // Test model
            var predictions = tf.equal(tf.argmax(model, 1), tf.argmax(mnistOutput, 1));
            // Calculate accuracy
            var acc = tf.reduce_mean(tf.cast(predictions, tf.float32));

            accuracy = acc.eval(session, (mnistShape, mnistData.Test.Data), (mnistOutput, mnistData.Test.Labels));
            print($"Accuracy: {accuracy:F4}");
        }
    }
}
