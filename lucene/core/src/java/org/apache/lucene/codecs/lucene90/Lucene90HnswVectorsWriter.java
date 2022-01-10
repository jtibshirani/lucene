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

package org.apache.lucene.codecs.lucene90;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.RandomAccessVectorValues;
import org.apache.lucene.index.RandomAccessVectorValuesProducer;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.NeighborArray;

/**
 * Writes vector values and knn graphs to index segments.
 *
 * @lucene.experimental
 */
public final class Lucene90HnswVectorsWriter extends KnnVectorsWriter {

  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta, vectorData, vectorIndex;

  private final int maxConn;
  private final int beamWidth;
  private boolean finished;

  Lucene90HnswVectorsWriter(SegmentWriteState state, int maxConn, int beamWidth)
      throws IOException {
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;

    assert state.fieldInfos.hasVectorValues();
    segmentWriteState = state;

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, Lucene90HnswVectorsFormat.META_EXTENSION);

    String vectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene90HnswVectorsFormat.VECTOR_DATA_EXTENSION);

    String indexDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene90HnswVectorsFormat.VECTOR_INDEX_EXTENSION);

    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorData = state.directory.createOutput(vectorDataFileName, state.context);
      vectorIndex = state.directory.createOutput(indexDataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          Lucene90HnswVectorsFormat.META_CODEC_NAME,
          Lucene90HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorData,
          Lucene90HnswVectorsFormat.VECTOR_DATA_CODEC_NAME,
          Lucene90HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorIndex,
          Lucene90HnswVectorsFormat.VECTOR_INDEX_CODEC_NAME,
          Lucene90HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public void writeField(FieldInfo fieldInfo, KnnVectorsReader knnVectorsReader)
      throws IOException {
    VectorValues vectors = knnVectorsReader.getVectorValues(fieldInfo.name);
    long pos = vectorData.getFilePointer();
    // write floats aligned at 4 bytes. This will not survive CFS, but it shows a small benefit when
    // CFS is not used, eg for larger indexes
    long padding = (4 - (pos & 0x3)) & 0x3;
    long vectorDataOffset = pos + padding;
    for (int i = 0; i < padding; i++) {
      vectorData.writeByte((byte) 0);
    }
    // TODO - use a better data structure; a bitset? DocsWithFieldSet is p.p. in o.a.l.index
    int[] docIds = new int[vectors.size()];
    int count = 0;
    for (int docV = vectors.nextDoc(); docV != NO_MORE_DOCS; docV = vectors.nextDoc(), count++) {
      // write vector
      writeVectorValue(vectors);
      docIds[count] = docV;
    }
    // count may be < vectors.size() e,g, if some documents were deleted
    long[] offsets = new long[count];
    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
    long vectorIndexOffset = vectorIndex.getFilePointer();
    if (vectors instanceof RandomAccessVectorValuesProducer) {
      writeGraph(
          vectorIndex,
          (RandomAccessVectorValuesProducer) vectors,
          fieldInfo.getVectorSimilarityFunction(),
          vectorIndexOffset,
          offsets,
          count,
          maxConn,
          beamWidth);
    } else {
      throw new IllegalArgumentException(
          "Indexing an HNSW graph requires a random access vector values, got " + vectors);
    }
    long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;
    writeMeta(
        fieldInfo,
        vectorDataOffset,
        vectorDataLength,
        vectorIndexOffset,
        vectorIndexLength,
        count,
        docIds);
    writeGraphOffsets(meta, offsets);
  }

  private void writeMeta(
      FieldInfo field,
      long vectorDataOffset,
      long vectorDataLength,
      long indexDataOffset,
      long indexDataLength,
      int size,
      int[] docIds)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    meta.writeVLong(indexDataOffset);
    meta.writeVLong(indexDataLength);
    meta.writeInt(field.getVectorDimension());
    meta.writeInt(size);
    for (int i = 0; i < size; i++) {
      // TODO: delta-encode, or write as bitset
      meta.writeVInt(docIds[i]);
    }
  }

  private void writeVectorValue(VectorValues vectors) throws IOException {
    // write vector value
    BytesRef binaryValue = vectors.binaryValue();
    assert binaryValue.length == vectors.dimension() * Float.BYTES;
    vectorData.writeBytes(binaryValue.bytes, binaryValue.offset, binaryValue.length);
  }

  private void writeGraphOffsets(IndexOutput out, long[] offsets) throws IOException {
    long last = 0;
    for (long offset : offsets) {
      out.writeVLong(offset - last);
      last = offset;
    }
  }

  private void writeGraph(
      IndexOutput graphData,
      RandomAccessVectorValuesProducer vectorValues,
      VectorSimilarityFunction similarityFunction,
      long graphDataOffset,
      long[] offsets,
      int count,
      int maxConn,
      int beamWidth)
      throws IOException {
    HnswGraphBuilder hnswGraphBuilder =
        new HnswGraphBuilder(
            vectorValues, similarityFunction, maxConn, beamWidth, HnswGraphBuilder.randSeed);
    hnswGraphBuilder.setInfoStream(segmentWriteState.infoStream);
    HnswGraph graph = hnswGraphBuilder.build(vectorValues.randomAccess());

    for (int ord = 0; ord < count; ord++) {
      // write graph
      offsets[ord] = graphData.getFilePointer() - graphDataOffset;

      NeighborArray neighbors = graph.getNeighbors(ord);
      int size = neighbors.size();

      // Destructively modify; it's ok we are discarding it after this
      int[] nodes = neighbors.node();
      Arrays.sort(nodes, 0, size);
      graphData.writeInt(size);

      int lastNode = -1; // to make the assertion work?
      for (int i = 0; i < size; i++) {
        int node = nodes[i];
        assert node > lastNode : "nodes out of order: " + lastNode + "," + node;
        assert node < offsets.length : "node too large: " + node + ">=" + offsets.length;
        graphData.writeVInt(node - lastNode);
        lastNode = node;
      }
    }
  }

  private VectorValues mergeVectorValues(FieldInfo mergeFieldInfo, MergeState mergeState) throws IOException {
    List<VectorValuesSub> subs = new ArrayList<>();
    int dimension = -1;
    VectorSimilarityFunction similarityFunction = null;
    int nonEmptySegmentIndex = 0;
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
      if (knnVectorsReader != null) {
        if (mergeFieldInfo != null && mergeFieldInfo.hasVectorValues()) {
          int segmentDimension = mergeFieldInfo.getVectorDimension();
          VectorSimilarityFunction segmentSimilarityFunction =
              mergeFieldInfo.getVectorSimilarityFunction();
          if (dimension == -1) {
            dimension = segmentDimension;
            similarityFunction = mergeFieldInfo.getVectorSimilarityFunction();
          } else if (dimension != segmentDimension) {
            throw new IllegalStateException(
                "Varying dimensions for vector-valued field "
                    + mergeFieldInfo.name
                    + ": "
                    + dimension
                    + "!="
                    + segmentDimension);
          } else if (similarityFunction != segmentSimilarityFunction) {
            throw new IllegalStateException(
                "Varying similarity functions for vector-valued field "
                    + mergeFieldInfo.name
                    + ": "
                    + similarityFunction
                    + "!="
                    + segmentSimilarityFunction);
          }
          VectorValues values = knnVectorsReader.getVectorValues(mergeFieldInfo.name);
          if (values != null) {
            subs.add(
                new VectorValuesSub(nonEmptySegmentIndex++, mergeState.docMaps[i], values));
          }
        }
      }
    }
    return new VectorValuesMerger(subs, mergeState);
  }

  @Override
  public void mergeVectors(FieldInfo mergeFieldInfo, final MergeState mergeState)
      throws IOException {
    if (mergeState.infoStream.isEnabled("VV")) {
      mergeState.infoStream.message("VV", "merging " + mergeState.segmentInfo);
    }
    // Create a new VectorValues by iterating over the sub vectors, mapping the resulting
    // docids using docMaps in the mergeState.
    writeField(
        mergeFieldInfo,
        new KnnVectorsReader() {
          @Override
          public long ramBytesUsed() {
            return 0;
          }

          @Override
          public void close() throws IOException {
            throw new UnsupportedOperationException();
          }

          @Override
          public void checkIntegrity() throws IOException {
            throw new UnsupportedOperationException();
          }

          @Override
          public VectorValues getVectorValues(String field) throws IOException {
            return mergeVectorValues(mergeFieldInfo, mergeState);
          }

          @Override
          public TopDocs search(String field, float[] target, int k, Bits acceptDocs)
              throws IOException {
            throw new UnsupportedOperationException();
          }
        });

    if (mergeState.infoStream.isEnabled("VV")) {
      mergeState.infoStream.message("VV", "merge done " + mergeState.segmentInfo);
    }
  }

  /** Tracks state of one sub-reader that we are merging */
  private static class VectorValuesSub extends DocIDMerger.Sub {

    final VectorValues values;
    final int segmentIndex;
    int count;

    VectorValuesSub(int segmentIndex, MergeState.DocMap docMap, VectorValues values) {
      super(docMap);
      this.values = values;
      this.segmentIndex = segmentIndex;
      assert values.docID() == -1;
    }

    @Override
    public int nextDoc() throws IOException {
      int docId = values.nextDoc();
      if (docId != NO_MORE_DOCS) {
        // Note: this does count deleted docs since they are present in the to-be-merged segment
        ++count;
      }
      return docId;
    }
  }

  /**
   * View over multiple VectorValues supporting iterator-style access via DocIdMerger. Maintains a
   * reverse ordinal mapping for documents having values in order to support random access by dense
   * ordinal.
   */
  private static class VectorValuesMerger extends VectorValues
      implements RandomAccessVectorValuesProducer {
    private final List<VectorValuesSub> subs;
    private final DocIDMerger<VectorValuesSub> docIdMerger;
    private final int[] ordBase;
    private final int cost;
    private int size;

    private int docId;
    private VectorValuesSub current;
    /* For each doc with a vector, record its ord in the segments being merged. This enables random
     * access into the unmerged segments using the ords from the merged segment.
     */
    private int[] ordMap;
    private int ord;

    VectorValuesMerger(List<VectorValuesSub> subs, MergeState mergeState) throws IOException {
      this.subs = subs;
      docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
      int totalCost = 0, totalSize = 0;
      for (VectorValuesSub sub : subs) {
        totalCost += sub.values.cost();
        totalSize += sub.values.size();
      }
      /* This size includes deleted docs, but when we iterate over docs here (nextDoc())
       * we skip deleted docs. So we sneakily update this size once we observe that iteration is complete.
       * That way by the time we are asked to do random access for graph building, we have a correct size.
       */
      cost = totalCost;
      size = totalSize;
      ordMap = new int[size];
      ordBase = new int[subs.size()];
      int lastBase = 0;
      for (int k = 0; k < subs.size(); k++) {
        int size = subs.get(k).values.size();
        ordBase[k] = lastBase;
        lastBase += size;
      }
      docId = -1;
    }

    @Override
    public int docID() {
      return docId;
    }

    @Override
    public int nextDoc() throws IOException {
      current = docIdMerger.next();
      if (current == null) {
        docId = NO_MORE_DOCS;
        /* update the size to reflect the number of *non-deleted* documents seen so we can support
         * random access. */
        size = ord;
      } else {
        docId = current.mappedDocID;
        ordMap[ord++] = ordBase[current.segmentIndex] + current.count - 1;
      }
      return docId;
    }

    @Override
    public float[] vectorValue() throws IOException {
      return current.values.vectorValue();
    }

    @Override
    public BytesRef binaryValue() throws IOException {
      return current.values.binaryValue();
    }

    @Override
    public RandomAccessVectorValues randomAccess() {
      return new VectorValuesMerger.MergerRandomAccess();
    }

    @Override
    public int advance(int target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public long cost() {
      return cost;
    }

    @Override
    public int dimension() {
      return subs.get(0).values.dimension();
    }

    class MergerRandomAccess implements RandomAccessVectorValues {

      private final List<RandomAccessVectorValues> raSubs;

      MergerRandomAccess() {
        raSubs = new ArrayList<>(subs.size());
        for (VectorValuesSub sub : subs) {
          if (sub.values instanceof RandomAccessVectorValuesProducer) {
            raSubs.add(((RandomAccessVectorValuesProducer) sub.values).randomAccess());
          } else {
            throw new IllegalStateException(
                "Cannot merge VectorValues without support for random access");
          }
        }
      }

      @Override
      public int size() {
        return size;
      }

      @Override
      public int dimension() {
        return VectorValuesMerger.this.dimension();
      }

      @Override
      public float[] vectorValue(int target) throws IOException {
        int unmappedOrd = ordMap[target];
        int segmentOrd = Arrays.binarySearch(ordBase, unmappedOrd);
        if (segmentOrd < 0) {
          // get the index of the greatest lower bound
          segmentOrd = -2 - segmentOrd;
        }
        while (segmentOrd < ordBase.length - 1 && ordBase[segmentOrd + 1] == ordBase[segmentOrd]) {
          // forward over empty segments which will share the same ordBase
          segmentOrd++;
        }
        return raSubs.get(segmentOrd).vectorValue(unmappedOrd - ordBase[segmentOrd]);
      }

      @Override
      public BytesRef binaryValue(int targetOrd) throws IOException {
        throw new UnsupportedOperationException();
      }
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;

    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
      CodecUtil.writeFooter(vectorIndex);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorData, vectorIndex);
  }
}
