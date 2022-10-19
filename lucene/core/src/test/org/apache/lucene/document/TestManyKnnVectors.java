/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.document;

import com.carrotsearch.randomizedtesting.annotations.TimeoutSuite;
import com.carrotsearch.randomizedtesting.generators.RandomPicks;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.Monster;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.bkd.BKDConfig;
import org.apache.lucene.util.bkd.BKDReader;
import org.apache.lucene.util.bkd.BKDWriter;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

// e.g. run like this: ant test -Dtestcase=Test4BBKDPoints -Dtests.nightly=true -Dtests.verbose=true
// -Dtests.monster=true
//
//   or: python -u /l/util/src/python/repeatLuceneTest.py -heap 4g -once -nolog -tmpDir /b/tmp
// -logDir /l/logs Test4BBKDPoints.test2D -verbose

@TimeoutSuite(millis = 7_200_000) // 2 hour timeout
@Monster("takes ~1 hour and needs 2GB heap")
public class TestManyKnnVectors extends LuceneTestCase {
  public void testLargeSegment() throws Exception {
    IndexWriterConfig iwc = newIndexWriterConfig();
    if (random().nextBoolean()) {
      iwc.setIndexSort(new Sort(new SortField("sortkey", SortField.Type.INT)));
    }
    String fieldName = "field";
    VectorSimilarityFunction similarityFunction = RandomPicks.randomFrom(random(), VectorSimilarityFunction.values());
    try (Directory dir = FSDirectory.open(createTempDir("ManyKnnVectors"));
         IndexWriter iw = new IndexWriter(dir, iwc)) {
      int numVectors = 600_000;
      int dims = VectorValues.MAX_DIMENSIONS;

      for (int i = 0; i < numVectors; i++) {
        float[] vector = randomVector(dims);
        Document doc = new Document();
        doc.add(new KnnVectorField(fieldName, vector, similarityFunction));
        doc.add(new NumericDocValuesField("sortkey", random().nextInt(100)));
        iw.addDocument(doc);
        if (VERBOSE && i % 10_000 == 0) {
          System.out.println("Indexed " + i + " vectors out of " + numVectors);
        }
      }
      iw.forceMerge(1);

      try (IndexReader reader = DirectoryReader.open(iw)) {
        assertEquals(1, reader.leaves().size());
        LeafReaderContext ctx = reader.leaves().get(0);

        VectorValues vectorValues = ctx.reader().getVectorValues(fieldName);
        assertNotNull(vectorValues);
        assertEquals(numVectors, vectorValues.size());

        int numVectorsRead = 0;
        while (vectorValues.nextDoc() != NO_MORE_DOCS) {
          float[] v = vectorValues.vectorValue();
          assertEquals(dims, v.length);
          numVectorsRead++;
        }
        assertEquals(numVectors, numVectorsRead);
      }
    }
  }

  private float[] randomVector(int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = random().nextFloat();
    }
    VectorUtil.l2normalize(v);
    return v;
  }
}
