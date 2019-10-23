"""Filter pipeline"""

import collections
import itertools
import logging
import sys

from tqdm import tqdm

from . import LengthRatioFilter, LanguageIDFilter, \
    LengthFilter, LongWordFilter, HtmlTagFilter, CharacterScoreFilter, \
    TerminalPunctuationFilter, NonZeroNumeralsFilter
from .lm import CrossEntropyFilter
from .word_alignment import WordAlignFilter


logger = logging.getLogger(__name__)


def grouper(iterable, n):
    """Split data into fixed-length chunks"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class FilterPipeline:
    """Pipeline for combining multiple filters"""

    def __init__(self, filters=None):
        self.filters = [] if filters is None else filters
        self._chunksize = 10000

    @classmethod
    def from_config(cls, config):
        """Initilize filter pipeline from configuration dictionary"""
        pipeline = cls()
        for f in config:
            name = next(iter(f.keys()))
            attributes = f[name]
            filter_ = getattr(sys.modules[__name__], name)
            pipeline.filters.append(filter_(**attributes))
        return pipeline

    def get_score_tuples(self):
        """Return unique score name tuples for the filters in the pipeline"""
        fnames = [(f.__class__.__name__, f.name) for f in self.filters]
        counts = collections.Counter(fnames)
        instances = collections.Counter()
        renamed = []
        for nametuple in fnames:
            clsname, name = nametuple
            if counts[nametuple] > 1:
                instances[nametuple] += 1
                newtuple = (clsname, str(instances[nametuple])) if name is None \
                           else (clsname, name, str(instances[nametuple]))
            else:
                newtuple = (clsname, ) if name is None else (clsname, name)
            renamed.append(newtuple)
        return renamed

    def score(self, pairs):
        """Yield dictionaries of filter scores for sentence pairs"""

        def update_score_dict(scored, namet, score):
            for key in namet[:-1]:
                if key not in scored:
                    scored[key] = {}
                scored = scored[key]
            scored[namet[-1]] = score

        fnames = self.get_score_tuples()
        for num, chunk in enumerate(grouper(pairs, self._chunksize)):
            chunk_scores = []
            for namet, filt in zip(fnames, self.filters):
                logger.info("Processing chunk %s with %s", num, '.'.join(namet))
                scorelist = list(tqdm(filt.score(chunk)))
                chunk_scores.append(scorelist)
            for scores in zip(*chunk_scores):
                output = {}
                for idx, score in enumerate(scores):
                    update_score_dict(output, fnames[idx], score)
                yield output

    def filter(self, pairs):
        """Yield sentence pairs accepted by all filters"""
        for f in self.filters:
            pairs = f.filter(pairs)
        return pairs
