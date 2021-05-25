package org.apache.lucene.search;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.util.VectorUtil;

/**
 * Note: copied from luceneutil with a few simplifications.
 */
public class KnnQuery extends Query {

    private final String field;
    private final float[] vector;
    private final int topK;
    private final int fanout;
    private final String text;

    public KnnQuery(String field, String text, float[] vector, int topK, int fanout) {
        this.field = field;
        this.text = text;
        this.vector = vector;
        this.topK = topK;
        this.fanout = fanout;
    }

    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        return new KnnWeight();
    }

    @Override
    public boolean equals(Object obj) {
        return sameClassAs(obj) &&
                ((KnnQuery) obj).field.equals(field) &&
                Arrays.equals(((KnnQuery) obj).vector, vector);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, Arrays.hashCode(vector));
    }

    @Override
    public String toString(String field) {
        return "<vector:knn:" + field + "<" + text + ">[" + vector[0] + ",...]>";
    }

    @Override
    public void visit(QueryVisitor visitor) {
    }

    class KnnWeight extends Weight {

        KnnWeight() {
            super(KnnQuery.this);
        }

        @Override
        public Scorer scorer(LeafReaderContext context) throws IOException {
            return new TopDocScorer(this, context.reader().searchNearestVectors(field, vector, topK, fanout));
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return true;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            VectorValues vectors = context.reader().getVectorValues(field);
            vectors.advance(doc);
            float score = VectorUtil.dotProduct(vector, vectors.vectorValue());
            return Explanation.match(0, "" + getQuery() + " in " + doc + " score " + score);
        }

        @Override
        public Matches matches(LeafReaderContext context, int doc) throws IOException {
            return MatchesUtils.MATCH_WITH_NO_TERMS;
        }

        @Override
        public String toString() {
            return "weight(" + KnnQuery.this + ")";
        }
    }

    static class TopDocScorer extends Scorer {

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
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return topDocs.scoreDocs.length;
            }
        }
    }
}
