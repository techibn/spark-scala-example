package org.example

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfter

class AppTest extends AnyFunSuite with BeforeAndAfter {

  var spark: SparkSession = _
  var testData: DataFrame = _

  // Set up a SparkSession before running the tests
  before {
    spark = SparkSession.builder()
      .appName("VectorAssemblerTest")
      .master("local[*]")
      .getOrCreate()

    // Create a test DataFrame
    val data = Seq((1, 2.0, 3.0), (4, 5.0, 6.0))
    val columns = Seq("col1", "col2", "col3")
    testData = spark.createDataFrame(data).toDF(columns: _*)
  }

  // Tear down the SparkSession after running the tests
  after {
    if (spark != null) {
      spark.stop()
    }
  }

  test("VectorAssembler should assemble columns correctly") {
    val assembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3"))
      .setOutputCol("features")

    val assembledData = assembler.transform(testData)

    // Perform assertions on the assembled data
    assert(assembledData.columns.contains("features"))
    assert(assembledData.select("features").first()(0).isInstanceOf[org.apache.spark.ml.linalg.Vector])
  }

  test("VectorAssembler should assemble columns correctly 2"){
    val dataset = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")

    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")

    val output = assembler.transform(dataset)
    println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(false)
  }
}
