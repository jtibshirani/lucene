package org.apache.lucene.search;

import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

public class KnnVectorQuery2 extends Query {
    private final String field;
    private final float[] target;
    private final int k;
    private final int numCands;

    /**
     * Find the <code>k</code> nearest documents to the target vector according to the vectors in the
     * given field. <code>target</code> vector.
     *
     * @param field a field that has been indexed as a {@link KnnVectorField}.
     * @param target the target of the search
     * @param k the number of documents to find
     * @throws IllegalArgumentException if <code>k</code> is less than 1
     */
    public KnnVectorQuery2(String field, float[] target, int k, int numCands) {
        this.field = field;
        this.target = target;
        this.k = k;
        this.numCands = numCands;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        return new Weight(this) {

            @Override
            public Explanation explain(LeafReaderContext context, int doc) throws IOException {
                VectorValues vectors = context.reader().getVectorValues(field);
                vectors.advance(doc);
                float score = VectorUtil.squareDistance(target, vectors.vectorValue());
                return Explanation.match(0, "" + getQuery() + " in " + doc + " score " + score);
            }

            @Override
            public Scorer scorer(LeafReaderContext context) throws IOException {
                LeafReader reader = context.reader();
                TopDocs topDocs = reader.searchNearestVectors(field, target, numCands, reader.getLiveDocs());
                if (topDocs.scoreDocs.length > k) {
                    topDocs = new TopDocs(topDocs.totalHits, Arrays.copyOf(topDocs.scoreDocs, k));
                }
                return new TopDocScorer(this, topDocs);
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return true;
            }
        };
    }

    @Override
    public String toString(String field) {
        return getClass().getSimpleName() + ":" + this.field + "[" + target[0] + ",...][" + k + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object obj) {
        return sameClassAs(obj)
                && ((KnnVectorQuery2) obj).k == k
                && ((KnnVectorQuery2) obj).field.equals(field)
                && Arrays.equals(((KnnVectorQuery2) obj).target, target);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, k, Arrays.hashCode(target));
    }

    private static class TopDocScorer extends Scorer {

        private int upTo = -1;
        private final TopDocs topDocs;
        private final TopDocsIterator iterator;

        TopDocScorer(Weight weight, TopDocs topDocs) {
            super(weight);
            this.topDocs = topDocs;
            iterator = new TopDocsIterator();
        }

        @Override
        public int docID() {
            return iterator.docID();
        }

        @Override
        public float score() {
            return topDocs.scoreDocs[upTo].score;
        }

        @Override
        public float getMaxScore(int upTo) {
            throw new UnsupportedOperationException();
        }

        @Override
        public DocIdSetIterator iterator() {
            return iterator;
        }

        class TopDocsIterator extends DocIdSetIterator {
            @Override
            public int docID() {
                if (upTo < 0) {
                    return -1;
                }
                return topDocs.scoreDocs[upTo].doc;
            }

            @Override
            public int nextDoc() {
                if (++upTo >= topDocs.scoreDocs.length) {
                    return NO_MORE_DOCS;
                }
                return docID();
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }

            @Override
            public long cost() {
                return topDocs.scoreDocs.length;
            }
        }
    }
}