package test;


import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;

/**
 * Hello world!
 */
public class NumpyApp {
    private static final Logger log = LoggerFactory.getLogger(NumpyApp.class);

    public static void main(String[] args) throws IOException {
        Context context = Context.newBuilder().allowIO(true).allowAllAccess(true).build();
        log.info("started");

        Value bindings = context.getBindings("python");
        Double[][] embeddings = readEmbeddings();
        bindings.putMember("embeddings", embeddings);

        final URL pythonCode = NumpyApp.class.getClassLoader().getResource("numpy_test.py");
        context.eval(Source.newBuilder("python", pythonCode).build());

        test(context);
    }

    private static void test(Context context) {
        log.info("Test started");
        long startTime = System.currentTimeMillis();
        context.eval("python", "test()");
        log.info("Test run time {}", System.currentTimeMillis() - startTime);
    }


    private static Double[][] readEmbeddings() {
        Double[][] embeddings = new Double[10001][512];
        int i = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/embeddings.txt"))) {
            String line = reader.readLine();
            while (line != null) {
                Double[] embedding = Arrays.stream(line.split(",")).map(Double::parseDouble).toArray(Double[]::new);
                embeddings[i] = embedding;
                i++;
                line = reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return embeddings;
    }
}
