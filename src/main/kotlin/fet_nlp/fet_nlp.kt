package fet_nlp

import com.aliasi.classify.Classification
import com.aliasi.classify.Classified
import com.aliasi.classify.DynamicLMClassifier
import com.aliasi.util.Files
import opennlp.tools.doccat.DoccatFactory
import opennlp.tools.doccat.DoccatModel
import opennlp.tools.doccat.DocumentCategorizerME
import opennlp.tools.doccat.DocumentSampleStream
import opennlp.tools.namefind.NameFinderME
import opennlp.tools.namefind.NameSampleDataStream
import opennlp.tools.namefind.TokenNameFinderFactory
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.tokenize.TokenizerME
import opennlp.tools.tokenize.TokenizerModel
import opennlp.tools.util.InputStreamFactory
import opennlp.tools.util.PlainTextByLineStream
import opennlp.tools.util.TrainingParameters
import java.io.*
import java.lang.Exception
import java.nio.charset.Charset
import java.nio.file.Paths
import java.util.*


fun main(args: Array<String>) {
//    println("Tokenize part.")
//    openNLPNERExample()
//    testModel(getTrainModel(getTrainingText()))

//    println("Categorize part.")
//    generateTrainingData()
//    trainingOpenNLPClassificationModel()
//    useOpenNLPClassificationModel()

    trainSentimentAnalysisClassifier()
    useSentimentAnalysisClassifier()
}

// Tokenize part.
private val SENTENCES = listOf("Jack was taller than Peter. ",
        "However, Mr. Smith was taller than both of them. ",
        "The same could be said for Mary and Tommy. ",
        "Mary Anne was the tallest."
)

private fun getTrainingText(): String {

    val names = listOf("Bill", "Sue", "Mary Anne", "John Henry", "Patty",
            "Jones", "Smith", "Albertson", "Henry", "Robertson", "Peter"
    )
    val prefixes = listOf("", "Mr. ", "", "Dr. ", "", "Mrs. ", "", "Ms. ")
    val namesLen = names.size
    val prefixesLen = prefixes.size
    val nameIndex = Random()
    val prefixIndex = Random()
    val sentences = StringBuilder()
    val markPrefix = "<START:PERSON> "
    val markSuffix = " <END>"
    (0..15000).forEach {
        sentences.appendln("$markPrefix" +
                "${prefixes[prefixIndex.nextInt(prefixesLen)]}${names[nameIndex.nextInt(namesLen)]}" +
                "$markSuffix is " +
                "$markPrefix" +
                "${prefixes[prefixIndex.nextInt(prefixesLen)]}${names[nameIndex.nextInt(namesLen)]}" +
                "$markSuffix cousin.\nAlso, " +
                "$markPrefix" +
                "${prefixes[prefixIndex.nextInt(prefixesLen)]}${names[nameIndex.nextInt(namesLen)]}" +
                "$markSuffix is a parent.\n" +
                "It turns out that $markPrefix" +
                "${prefixes[prefixIndex.nextInt(prefixesLen)]}${names[nameIndex.nextInt(namesLen)]}" +
                "$markSuffix and " +
                "$markPrefix" +
                "${prefixes[prefixIndex.nextInt(prefixesLen)]}${names[nameIndex.nextInt(namesLen)]}" +
                "$markSuffix are not related.\n" +
                "$markPrefix" +
                "${prefixes[prefixIndex.nextInt(prefixesLen)]}${names[nameIndex.nextInt(namesLen)]}" +
                "$markSuffix comes from Canada.\n"

        )
    }
    return sentences.toString()
}

private fun getTrainModel(trainText: String): TokenNameFinderModel {
    return NameFinderME.train(
            "en",
            "person",
            NameSampleDataStream(
                    PlainTextByLineStream(
                            InputStreamFactory { ByteArrayInputStream(trainText.toByteArray()) },
                            Charset.forName("UTF-8")
                    )
            ),
            TrainingParameters.defaultParams(),
            TokenNameFinderFactory()
    )
}

private fun testModel(tokenNameFinderModel: TokenNameFinderModel) {
    println("openNLP testModel Example")
    val path = Paths.get("").toAbsolutePath().toString()
    try {
        FileInputStream(File("$path//data//en-token.zip")).use { tokenModelStream ->
            processData(tokenModelStream, tokenNameFinderModel)
        }
    } catch (ex: Exception) {
        ex.printStackTrace()
    }
}

private fun openNLPNERExample() {
    println("openNLP NER Example")
    try {
        val path = Paths.get("").toAbsolutePath().toString()
        FileInputStream(File("$path//data//en-token.zip")).use { tokenModelStream ->
            FileInputStream(File("$path//data//en-ner-person.zip")).use { nerModelInputStream ->
                processData(tokenModelStream, TokenNameFinderModel(nerModelInputStream))
            }
        }
    } catch (ex: Exception) {
        ex.printStackTrace()
    }
}

private fun processData(tokenModelStream: FileInputStream, nameModel: TokenNameFinderModel) {
    val tokenizerModel = TokenizerModel(tokenModelStream)
    val tokenizer = TokenizerME(tokenizerModel)
    val nameFinder = NameFinderME(nameModel)

    SENTENCES.forEach {
        println("Sentence: [$it]")
        val tokens = tokenizer.tokenize(it)
        val nameSpans = nameFinder.find(tokens)
        val spanProbabilities = nameFinder.probs(nameSpans)

        nameSpans.forEachIndexed { i, span ->
            println("Span: $span")
            if ((span.end - span.start) == 1) println("Entity: ${tokens[span.start]}")
            else println("Entity: ${tokens[span.start]} ${tokens[span.end - 1]}")
            println("Probability: ${spanProbabilities[i]}")
        }
        println()
    }
}


// Categorize part.
private fun generateTrainingData() {
    println("Generate training data.")
    try {
        val path = Paths.get("").toAbsolutePath().toString()
        FileOutputStream(File("$path//data//en-vehicle.train")).use {
            val out = BufferedOutputStream(it)
            val engines = listOf("", "V8", "V6", "4")
            val vins = listOf("", "v1203", "v0093", "v11", "v203")
            val available = listOf("", "na")

            val makeRNG = Random()
            val engineRNG = Random()
            val vinRNG = Random()
            val naRNG = Random()
            (0..50000).forEach {
                when (makeRNG.nextInt(3)) {
                    0 -> out.write(("100 " +
                            available[naRNG.nextInt(available.size)] + " Ford " +
                            available[naRNG.nextInt(available.size)] + " " +
                            engines[engineRNG.nextInt(engines.size)] + " " +
                            vins[vinRNG.nextInt(vins.size)]
                            ).toByteArray())
                    1 -> out.write(("200 " +
                            available[naRNG.nextInt(available.size)] + " Toyota " +
                            available[naRNG.nextInt(available.size)] + " " +
                            engines[engineRNG.nextInt(engines.size)] + " " +
                            vins[vinRNG.nextInt(vins.size)]
                            ).toByteArray())
                    2 -> out.write(("300 " +
                            available[naRNG.nextInt(available.size)] + " Honda " +
                            available[naRNG.nextInt(available.size)] + " " +
                            engines[engineRNG.nextInt(engines.size)] + " " +
                            vins[vinRNG.nextInt(vins.size)]
                            ).toByteArray())
                }
                out.write((engines[engineRNG.nextInt(engines.size)] + "\n").toByteArray())
            }
        }
    } catch (ex: IOException) {
        ex.printStackTrace()
    }
}

private fun trainingOpenNLPClassificationModel() {
    println("Training model.")
    try {
        val path = Paths.get("").toAbsolutePath().toString()
        FileInputStream(File("$path//data//en-vehicle.train")).use { trainingDataStream ->
            FileOutputStream(File("$path//data/en-vehicle.model")).use { modelOutStream ->
                val lineStream = PlainTextByLineStream(InputStreamFactory {trainingDataStream}, "UTF-8")
                val docSampleStream = DocumentSampleStream(lineStream)
                val model = DocumentCategorizerME.train(
                        "en",
                        docSampleStream,
                        TrainingParameters.defaultParams(),
                        DoccatFactory()
                )
                val modelOut = BufferedOutputStream(modelOutStream)
                model.serialize(modelOut)
            }
        }
    } catch (ex: IOException) {
        ex.printStackTrace()
    }
}

private fun useOpenNLPClassificationModel() {
    val path = Paths.get("").toAbsolutePath().toString()
    FileInputStream(File("$path//data//en-vehicle.model")).use { modelInputStream ->
        val categorizer = DocumentCategorizerME(DoccatModel(modelInputStream))
        val outcomes = categorizer.categorize(arrayOf("Toyota", "V6", "v234"))
        println("Outcomes size: ${outcomes.size}")
        println("Categories: (${categorizer.numberOfCategories})")
        (0..(categorizer.numberOfCategories - 1)).forEach {
            println("${it + 1}) Category: ${categorizer.getCategory(it)} - ${outcomes[it]}")
        }
        println("\nBest category: ${categorizer.getBestCategory(outcomes)}")
    }
}


// Classification part.
private val CATEGORIES = listOf("neg", "pos").toTypedArray()
private val NGRAMSIZE = 6
private val classifier = DynamicLMClassifier.createNGramProcess(
        CATEGORIES,
        NGRAMSIZE
)

private fun trainSentimentAnalysisClassifier() {
    println("Training the Sentiment Analysis Classifier...")
    val path = Paths.get("").toAbsolutePath().toString()
    val trainingDirectory = File("$path//data//txt_sentoken")
    CATEGORIES.forEach { category ->
        val classification = Classification(category)
        val file = File(trainingDirectory, category)
        val trainingFiles = file.listFiles()
        trainingFiles.forEach { trainingFile ->
            try{
                val review = Files.readFromFile(trainingFile, "ISO-8859-1")
                val classified = Classified<CharSequence>(review, classification)
                classifier.handle(classified)
            } catch (ex: IOException) {
                ex.printStackTrace()
            }
        }
    }
}

private fun useSentimentAnalysisClassifier() {
    println("Make classification...")
    var review = ""
    val path = Paths.get("").toAbsolutePath().toString()
    try {
        BufferedReader(FileReader("$path//data//review.txt")).use { reviewReader ->
            val sb = StringBuilder()
            reviewReader.lineSequence().forEach { line ->
                sb.append(line).append(" ")
            }
            review = sb.toString()
        }
    } catch(ex: IOException) {
        ex.printStackTrace()
    }
    println("Text: $review")
    val classification = classifier.classify(review)
    println("Best Category: ${classification.bestCategory()}")
}