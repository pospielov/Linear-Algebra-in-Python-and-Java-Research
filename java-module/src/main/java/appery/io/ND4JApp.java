package appery.io;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.function.Function;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 */
public class ND4JApp {
    private static final Logger log = LoggerFactory.getLogger(ND4JApp.class);

    public static void main(String[] args) throws IOException {
        log.info("started");

        INDArray embeddings = readEmbeddings();
        INDArray embeddings1 = embeddings.get(NDArrayIndex.point(0), NDArrayIndex.all());
        INDArray embeddings2 = embeddings.get(NDArrayIndex.interval(1,10001), NDArrayIndex.all());
        testCached(embeddings1, embeddings2);
    }

    private static void test(INDArray embeddings1, INDArray embeddings2) {
        log.info("Test started");
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < 100; i++) {
            //testLoop(embeddings1, embeddings2);
            testTransforms(embeddings1.dup(), embeddings2.dup());
        }

        log.info("Test run time {}", System.currentTimeMillis() - startTime);
    }


    private static void testCached(INDArray embeddings1, INDArray embeddings2) {
        log.info("Test started");
        INDArray embeddings2Norm = embeddings2.norm2(1);
        embeddings2 = embeddings2.transposei().divi(embeddings2Norm).transposei();
        INDArray embeddings2Copy = embeddings2.dup();
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < 1000; i++) {
            embeddings2Copy.assign(embeddings2);
            testTransformsCached(embeddings1.dup(), embeddings2Copy);
        }

        log.info("Test run time {}", System.currentTimeMillis() - startTime);
    }


    private static double testTransformsCached(INDArray embeddings1, INDArray embeddings2) {
        INDArray embeddings1Norm = embeddings1.norm2();
        embeddings1 = embeddings1.divi(embeddings1Norm);

        embeddings2 = embeddings2.subi(embeddings1);
        INDArray dist = embeddings2.norm2(1);

        // log.info("Distance: {}", dist.max());
        // log.info("ArgMax: {}", dist.argMax());
        return dist.max().getDouble(0);
        // Test run time 1019
    }


    private static double testTransforms(INDArray embeddings1, INDArray embeddings2) {
        INDArray embeddings1Norm = embeddings1.norm2();
        // Test run time 1902
        embeddings1 = embeddings1.divi(embeddings1Norm);
        // Test run time 1928

        INDArray embeddings2Norm = embeddings2.norm2(1);
        // Test run time 2143
        embeddings2 = embeddings2.transposei().divi(embeddings2Norm).transposei();
        // Test run time 2526

        embeddings2.subi(embeddings1);
        // Test run time 2764
        INDArray dist = embeddings2.norm2(1);
        // Test run time 3008

        // log.info("Distance: {}", dist.max());
        // log.info("ArgMax: {}", dist.argMax());
        return dist.max().getDouble(0);
        // Test run time 3099
    }

    private static double testTransformsOld(INDArray embeddings1, INDArray embeddings2) {
        INDArray embeddings1Norm = embeddings1.norm2();
        // Test run time 26
        embeddings1 = embeddings1.div(embeddings1Norm);
        // Test run time 64

        INDArray embeddings2Norm = embeddings2.norm2(1);
        // Test run time 281
        embeddings2 = embeddings2.transpose().divRowVector(embeddings2Norm).transpose();
        // Test run time 4040

        INDArray diff = embeddings1.sub(embeddings2);
        // Test run time 6016
        INDArray dist = Transforms.pow(diff, 2).sum(1);
        // Test run time 10356

        log.info("Distance: {}", dist.max());
        log.info("ArgMax: {}", dist.argMax());
        return dist.max().getDouble(0);
        // Test run time 10132

    }


    private static double testLoop(INDArray embeddings1, INDArray embeddings2) {
        double max = 0.0d;
        int index = -1;
        INDArray embeddings1Norm = embeddings1.div(embeddings1.norm2());
        for (int i = 0; i < 10000 ; i++) {
            INDArray embeddings2i = embeddings2.get(NDArrayIndex.point(i), NDArrayIndex.all());
            INDArray embeddings2iNorm = embeddings2i.div(embeddings2i.norm2());
            double distance = embeddings1Norm.distance2(embeddings2iNorm);
            if (max < distance) {
                max = distance;
                index = i;
            }
        }
        //log.info("Distance: {}", max);
        //log.info("Index: {}", index);
        return max;
    }


    private static INDArray readEmbeddings() {
        double[][] embeddings = new double[10001][512];
        int i = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/embeddings.txt"))) {
            String line = reader.readLine();
            while (line != null) {
                Double[] embedding = Arrays.stream(line.split(",")).map(Double::parseDouble).toArray(Double[]::new);
                embeddings[i] = ArrayUtils.toPrimitive(embedding);
                i++;
                line = reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return Nd4j.create(embeddings);
    }
}
