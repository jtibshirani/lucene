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

package org.apache.lucene.search;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.VectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import py4j.GatewayServer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class PythonEntryPoint {
    private static final String INDEX_NAME = "vector-index";
    private static final String ID_FIELD = "id";
    private static final String VECTOR_FIELD = "vector";

    private Directory directory;
    private IndexWriter indexWriter;

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new PythonEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }

    public void prepareIndex() throws IOException {
        Path path = Files.createTempDirectory(INDEX_NAME);
        directory = MMapDirectory.open(path);

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        iwc.setCodec(Codec.forName("Lucene90"));
        indexWriter = new IndexWriter(directory, iwc);
    }

    public void indexBatch(int startId, byte[] data) throws IOException {
        float[][] vectors = deserializeMatrix(data);

        int id = startId;
        for (float[] vector : vectors) {
            Document doc = new Document();
            doc.add(new StoredField(ID_FIELD, id++));

            FieldType vectorFieldType = VectorField.createHnswType(vector.length,
                    VectorValues.SimilarityFunction.EUCLIDEAN, 16, 500);
            doc.add(new VectorField(VECTOR_FIELD, vector, vectorFieldType));
            indexWriter.addDocument(doc);
        }
    }

    public void mergeAndCommit() throws IOException {
        indexWriter.forceMerge(1);
        indexWriter.close();
    }

    public List<List<Integer>> search(byte[] data, int k, int numCands) throws IOException {
        float[][] queryVectors = deserializeMatrix(data);
        List<List<Integer>> results = new ArrayList<>();

        IndexReader indexReader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(indexReader);
        for (float[] queryVector : queryVectors) {
            Query query = new KnnQuery(VECTOR_FIELD, "dummy text", queryVector, k, numCands - k);

            TopDocs topDocs = searcher.search(query, k);

            List<Integer> result = new ArrayList<>(k);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = indexReader.document(scoreDoc.doc);
                IndexableField field = doc.getField(ID_FIELD);

                assert field != null;
                result.add(field.numericValue().intValue());
            }
            results.add(result);
        }
        
        indexReader.close();
        return results;
    }

    public float[][] deserializeMatrix(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        int n = buffer.getInt(), m = buffer.getInt();
        float[][] matrix = new float[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                matrix[i][j] = buffer.getFloat();
            }
        }

        return matrix;
    }
}
