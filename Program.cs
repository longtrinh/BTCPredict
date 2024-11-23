using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace BTCPredict
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "btcusd_1-min_data.csv"; // Đường dẫn file dữ liệu
            var context = new MLContext();

            Console.WriteLine("Loading data...");
            var dataView = LoadData(context, dataPath);

            Console.WriteLine("Training model...");
            var model = TrainModel(context, dataView);

            Console.WriteLine("Making predictions...");
            MakePredictions(context, model, dataView);
        }

        static IDataView LoadData(MLContext context, string dataPath)
        {
            // Định nghĩa schema dữ liệu
            var loader = context.Data.CreateTextLoader(
                new TextLoader.Options
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("Timestamp", DataKind.Single, 0),
                        new TextLoader.Column("Open", DataKind.Single, 1),
                        new TextLoader.Column("High", DataKind.Single, 2),
                        new TextLoader.Column("Low", DataKind.Single, 3),
                        new TextLoader.Column("Close", DataKind.Single, 4),
                        new TextLoader.Column("Volume", DataKind.Single, 5)
                    }
                });

            return loader.Load(dataPath);
        }

        static ITransformer TrainModel(MLContext context, IDataView dataView)
        {
            // Sử dụng chỉ số "Close" để dự đoán
            var pipeline = context.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedClose",
                inputColumnName: "Close",
                windowSize: 10, // Số lượng giá trị trước đó để xem xét
                seriesLength: 30, // Độ dài chuỗi dữ liệu
                trainSize: 1000, // Kích thước dữ liệu huấn luyện
                horizon: 5, // Số bước dự đoán
                confidenceLevel: 0.95f, // Độ tin cậy
                confidenceLowerBoundColumn: "LowerBoundClose",
                confidenceUpperBoundColumn: "UpperBoundClose"
            );

            return pipeline.Fit(dataView);
        }

        static void MakePredictions(MLContext context, ITransformer model, IDataView dataView)
        {
            var forecastingEngine = model.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);

            var predictions = forecastingEngine.Predict();

            Console.WriteLine("Forecasting results:");
            for (int i = 0; i < predictions.ForecastedClose.Length; i++)
            {
                Console.WriteLine($"Day {i + 1}: Predicted Close = {predictions.ForecastedClose[i]:0.00} USD");
            }
        }

        // Định nghĩa input và output dữ liệu
        class ModelInput
        {
            public float Close { get; set; }
        }

        class ModelOutput
        {
            public float[] ForecastedClose { get; set; }
            public float[] LowerBoundClose { get; set; }
            public float[] UpperBoundClose { get; set; }
        }
    }
}
