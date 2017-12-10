package fet_nlp

import opennlp.tools.namefind.NameFinderME
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.tokenize.TokenizerME
import opennlp.tools.tokenize.TokenizerModel
import java.io.File
import java.io.FileInputStream
import java.lang.Exception


fun main(args: Array<String>) {
    openNLPNERExample()
}

val SENTENCES = listOf("Jack was taller than Peter. ",
        "Howewer, Mr. Smith was taller than both of them. ",
        "The same could be said for Mary and Tommy. ",
        "Mary Anne was the tallest."
)

fun openNLPNERExample() {
    println("openNLP NER Example")
    try {
        FileInputStream(File("c://code//openNlp//en-token.zip")).use { tokenModelStream ->
            FileInputStream(File("c://code//openNlp//en-ner-person.zip")).use { nerModelInputStream ->
                val tokenizerModel = TokenizerModel(tokenModelStream)
                val tokenizer = TokenizerME(tokenizerModel)
                val nameModel = TokenNameFinderModel(nerModelInputStream)
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
        }
    } catch (ex: Exception) {
        ex.printStackTrace()
    }

}
