import os
import unittest
from unittest import mock
import io
import sys
import xml.parsers.expat
import gzip
import shutil
import zipfile
import tempfile
import bz2

from opustools_pkg import OpusRead, OpusCat, OpusGet
from opustools_pkg.opus_read import AlignmentParserError
from opustools_pkg.parse.sentence_parser import SentenceParserError

def pairPrinterToVariable(**kwargs):
    old_stdout = sys.stdout
    printout = io.StringIO()
    sys.stdout = printout
    oprinter = OpusRead(**kwargs)
    oprinter.printPairs()
    oprinter.par.closeFiles()
    sys.stdout = old_stdout
    return printout.getvalue()

def add_to_root_dir(corpus=None, source=None, target=None,
        version='latest', preprocess=None, root_dir=None):

    OpusGet(directory=corpus, source=source, target=target, release=version,
        preprocess=preprocess, download_dir=root_dir, suppress_prompts=True,
        ).get_files()

    source_zip = '{corpus}_{version}_{preprocess}_{source}.zip'.format(
        corpus=corpus, version=version, preprocess=preprocess, source=source)
    os.rename(os.path.join(root_dir, source_zip),
        os.path.join(root_dir, corpus, version, preprocess, source+'.zip'))

    target_zip = '{corpus}_{version}_{preprocess}_{target}.zip'.format(
        corpus=corpus, version=version, preprocess=preprocess, target=target)
    os.rename(os.path.join(root_dir,target_zip),
        os.path.join(root_dir, corpus, version, preprocess, target+'.zip'))

    alignment_xml = ('{corpus}_{version}_{preprocess}_{source}-'
        '{target}.xml.gz').format(corpus=corpus, version=version,
            preprocess='xml', source=source, target=target)
    os.rename(os.path.join(root_dir, alignment_xml),
        os.path.join(root_dir, corpus, version, 'xml',
            source+'-'+target+'.xml.gz'))

class TestOpusRead(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if ('OPUSREAD_TEST_TEMPDIR' in os.environ.keys() and
            os.path.exists(os.environ['OPUSREAD_TEST_TEMPDIR'])):
                self.tempdir1 = os.environ['OPUSREAD_TEST_TEMPDIR']
        else:
            self.tempdir1 = tempfile.mkdtemp()
            os.mkdir(os.path.join(self.tempdir1, 'test_files'))

            os.makedirs(os.path.join(self.tempdir1, 'RF', 'xml', 'en'))
            with open(os.path.join(self.tempdir1, 'RF', 'xml', 'en',
                    '1996.xml'), 'w') as f:
                f.write('<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<text>'
                '<head>\n<meta id="1"> \n <w id="w1.1">The</w> \n <w id="'
                'w1.2">Hound</w> \n <w id="w1.3">of</w> \n <w id="w1.4">the'
                '</w> \n <w id="w1.5">Baskervilles</w>   \n <w id="w1.6">by'
                '</w> \n <w id="w1.7">Sir</w> \n <w id="w1.8">Arthur</w> \n '
                '<w id="w1.9">Conan</w> \n <w id="w1.10">Doyle</w>   \n <w '
                'id="w1.11">Aligned</w> \n <w id="w1.12">by</w>\n <w id="w1.'
                '13">:</w> \n <w id="w1.14">András</w> \n <w id="w1.15">'
                'Farkas</w> \n <w id="w1.16">(</w>\n <w id="w1.17">fully</w> '
                '\n <w id="w1.18">reviewed</w>\n <w id="w1.19">)</w> \n</'
                'meta></head><body>\n<s cld2="en" cld2conf="0.97" id="s1" '
                'langid="en" langidconf="0.99">\n <chunk id="c1-1" type="NP'
                '">\n  <w hun="NN" id="w1.1" lem="source" pos="NN" tree="NN'
                '">Source&amp;&lt;&gt;"\'</w>\n </chunk>\n <w hun=":" '
                'id="w1.2" lem=":" pos'
                '=":" tree=":">:</w>\n <chunk id="c1-3" type="NP">\n  <w '
                'hun="NNP" id="w1.3" pos="NNP" tree="NN">manybooks.'
                'netAudiobook</w>\n  <w hun="JJ" id="w1.4" lem="available" '
                'pos="NN" tree="JJ">available</w>\n </chunk>\n <chunk id="'
                'c1-4" type="ADVP">\n  <w hun="RB" id="w1.5" lem="here" '
                'pos="RB" tree="RB">here</w>\n </chunk>\n</s>\n\n\n\n<s '
                'cld2="un" cld2conf="0.0" id="s4" langid="en" langidconf'
                '="0.17">\n <chunk id="c4-1" type="NP">\n  <w hun="NNP" '
                'id="w4.1" lem="Chapter" pos="NNP" tree="NP">Chapter</w>\n  '
                '<w hun="CD" id="w4.2" lem="1" pos="CD" tree="CD">1</w>\n  '
                '<w hun="NNP" id="w4.3" lem="Mr." pos="NNP" tree="NP">Mr.</'
                'w>\n  <w hun="NNP" id="w4.4" lem="Sherlock" pos="NNP" tree'
                '="NP">Sherlock</w>\n  <w hun="NNP" id="w4.5" lem="Holmes" '
                'pos="NNP" tree="NP">Holmes</w>\n </chunk>\n</s><p id="p5'
                '">\n<s cld2="en" cld2conf="0.99" id="s5.0" langid="en" '
                'langidconf="1.0">\n <chunk id="c5.0-1" type="NP">\n  <w hun'
                '="NNP" id="w5.0.1" lem="Mr." pos="NNP" tree="NP">Mr.</w>\n  '
                '<w hun="NNP" id="w5.0.2" lem="Sherlock" pos="NNP" tree="NP">'
                'Sherlock</w>\n  <w hun="NNP" id="w5.0.3" lem="Holmes" pos="'
                'NNP" tree="NP">Holmes</w>\n</chunk>\n</s>\n\n\n<s cld2="un" '
                'cld2conf="0.0" id="s8.1" langid="en" langidconf="0.17">\n '
                '<chunk id="c8.1-1" type="NP">\n  <w hun="PRP" id="w8.1.1" '
                'lem="I" pos="PRP" tree="PP">I</w>\n </chunk>\n <chunk id="c8'
                '.1-2" type="VP">\n  <w hun="VBP" id="w8.1.2" lem="believe" '
                'pos="VBP" tree="VVP">believe</w>\n </chunk>\n</s></p>\n\n<p '
                'id="p167">\n<s cld2="un" cld2conf="0.0" id="s167.0" langid="'
                'de" langidconf="0.47">\n <chunk id="c167.0-1" type="NP">\n  '
                '<w hun="JJ" id="w167.0.1" lem="&quot;" pos="NN" tree="``">"</'
                'w>\n  <w hun="NN" id="w167.0.2" lem="excellent" pos="NNP" '
                'tree="JJ">Excellent</w>\n </chunk>\n <w hun="." id="w167.0.'
                '3" lem="!" pos="." tree="SENT">!</w>\n</s>\n \n\n\n</p>\n '
                '</body>\n</text>\n')

            with zipfile.ZipFile(os.path.join(self.tempdir1,
                    'RF_v1_xml_en.zip'), 'w') as zf:
                zf.write(os.path.join(self.tempdir1, 'RF', 'xml', 'en',
                    '1996.xml'), arcname='RF/xml/en/1996.xml')

            os.mkdir(os.path.join(self.tempdir1, 'RF', 'xml', 'sv'))
            with open(os.path.join(self.tempdir1, 'RF', 'xml', 'sv',
                    '1996.xml'), 'w') as f:
                f.write('<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<text'
                '>\n <head>\n  <meta> The Hound of the Baskervilles \n by '
                'Sir Arthur Conan Doyle \n Aligned by: András Farkas (fully '
                'reviewed) \n </meta>\n </head>\n <body>\n<s cld2="en" '
                'cld2conf="0.96" id="s1" langid="de" langidconf="0.66">\n '
                '<w id="w1.1">Source</w>\n <w id="w1.2">:</w> \n <w id="w1.'
                '3">Project</w> \n <w id="w1.4">Gutenberg</w>\n</s>\n\n<s '
                'cld2="ia" cld2conf="0.95" id="s4" langid="et" langidconf'
                '="0.42">\n <w id="w4.1">Herra</w> \n <w id="w4.2">Sherlock'
                '</w> \n <w id="w4.3">Holmes</w>\n <w id="w4.4">.</w>\n</s'
                '><p id="p5">\n<s cld2="fi" cld2conf="0.99" id="s5.0" langid'
                '="fi" langidconf="1.0">\n <w id="w5.0.1">Herra</w> \n <w '
                'id="w5.0.2">Sherlock</w> \n <w id="w5.0.3">Holmes</w>\n</'
                's>\n   \n<s cld2="fi" cld2conf="0.97" id="s8.1" langid="fi" '
                'langidconf="1.0">\n <w id="w8.1.1">Luulenpa</w> \n <w id="'
                'w8.1.2">että</w> \n <w id="w8.1.3">sinulla</w> \n</s></p>\n'
                '<p id="p167">\n<s cld2="un" cld2conf="0.0" id="s167.0" '
                'langid="fi" langidconf="0.38">\n <w id="w167.0.1">"</w>\n '
                '<w id="w167.0.2">Erinomaista</w>\n <w id="w167.0.3">.</w>\n'
                '</s></p>\n </body>\n</text>\n')

            with zipfile.ZipFile(os.path.join(self.tempdir1,
                    'RF_v1_xml_sv.zip'), 'w') as zf:
                zf.write(os.path.join(self.tempdir1, 'RF', 'xml', 'sv',
                    '1996.xml'), arcname='RF/xml/sv/1996.xml')

            shutil.copyfile(os.path.join(self.tempdir1, 'RF_v1_xml_en.zip'),
                os.path.join(self.tempdir1, 'en.zip'))
            shutil.copyfile(os.path.join(self.tempdir1, 'RF_v1_xml_sv.zip'),
                os.path.join(self.tempdir1, 'sv.zip'))

            with open(os.path.join(self.tempdir1, 'books_alignment.xml'),
                    'w') as f:
                f.write('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE '
                'cesAlign PUBLIC "-//CES//DTD XML cesAlign//EN" "">\n'
                '<cesAlign version="1.0">\n<linkGrp targType="s" fromDoc="en/'
                '1996.xml.gz" '
                'toDoc="sv/1996.xml.gz" '
                '>\n<link xtargets="s1;s1" id="SL1"/>\n<link xtargets="s4;s4" '
                'id="SL4"/>\n<link xtargets="s5.0;s5.0" id="SL5.0"/>\n<link '
                'xtargets="s8.1;s8.1" id="SL8.1"/>\n<link xtargets="s167.0'
                ';s167.0" id="SL167.0"/>\n  </linkGrp>\n</cesAlign>\n')

            with gzip.open(os.path.join(self.tempdir1,
                    'RF_v1_xml_en-sv.xml.gz'), 'wb') as f:
                with open(os.path.join(self.tempdir1, 'books_alignment.xml'),
                        'rb') as b:
                    f.write(b.read())

        if ('OPUSREAD_TEST_ROOTDIR' in os.environ.keys() and
            os.path.exists(os.environ['OPUSREAD_TEST_ROOTDIR'])):
            self.root_directory = os.environ['OPUSREAD_TEST_ROOTDIR']
        else:
            self.root_directory = tempfile.mkdtemp()

            os.makedirs(os.path.join(self.root_directory, 'RF', 'latest',
                'xml'))

            add_to_root_dir(corpus='RF', source='en', target='sv',
                preprocess='xml', root_dir=self.root_directory)

            add_to_root_dir(corpus='RF', source='en', target='es',
                preprocess='xml', root_dir=self.root_directory)

            os.mkdir(os.path.join(self.root_directory, 'RF', 'latest', 'raw'))

            add_to_root_dir(corpus='RF', source='en', target='sv',
                preprocess='raw', root_dir=self.root_directory)

            os.mkdir(os.path.join(self.root_directory, 'RF', 'latest',
                'parsed'))

            add_to_root_dir(corpus='RF', source='en', target='sv',
                preprocess='parsed', root_dir=self.root_directory)

            os.makedirs(os.path.join(self.root_directory, 'RF', 'v1', 'xml'))

            add_to_root_dir(corpus='RF', source='en', target='sv',
                version='v1', preprocess='xml', root_dir=self.root_directory)

            add_to_root_dir(corpus='RF', source='en', target='es',
                version='v1', preprocess='xml', root_dir=self.root_directory)

            os.makedirs(os.path.join(self.root_directory, 'OpenSubtitles',
                'latest', 'raw'))
            os.makedirs(os.path.join(self.root_directory, 'OpenSubtitles',
                'latest', 'xml'))

            add_to_root_dir(corpus='OpenSubtitles', source='eo', target='tl',
                preprocess='raw', root_dir=self.root_directory)

            add_to_root_dir(corpus='OpenSubtitles', source='eo', target='tl',
                preprocess='xml', root_dir=self.root_directory)

            os.makedirs(os.path.join(self.root_directory, 'Books',
                'latest', 'xml'))

            add_to_root_dir(corpus='Books', source='eo', target='pt',
                preprocess='xml', root_dir=self.root_directory)

            add_to_root_dir(corpus='RF', source='fr', target='sv',
                preprocess='xml', root_dir=self.root_directory)

            os.remove(os.path.join(self.root_directory, 'RF', 'latest', 'xml',
                'fr.zip'))

        self.opr = OpusRead(directory='RF', source='en', target='sv',
            root_directory=self.root_directory)
        self.opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.fastopr = OpusRead(directory='RF', source='en', target='sv',
            root_directory=self.root_directory, fast=True)
        self.fastopr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        self.maxDiff= None

    @classmethod
    def tearDownClass(self):
        self.opr.par.sPar.document.close()
        self.opr.par.tPar.document.close()
        self.opr.par.closeFiles()
        self.fastopr.par.sPar.document.close()
        self.fastopr.par.tPar.document.close()
        self.fastopr.par.closeFiles()
        if ('OPUSREAD_TEST_SAVE' in os.environ.keys() and
                os.environ['OPUSREAD_TEST_SAVE'] == 'true'):
            print('\nTEMPDIR:', self.tempdir1)
            print('ROOTDIR:', self.root_directory)
        else:
            shutil.rmtree(self.tempdir1)
            shutil.rmtree(self.root_directory)

    def tearDown(self):
        self.opr.par.write_mode='normal'
        self.opr.par.slim=['all']
        self.opr.par.tlim=['all']
        self.opr.par.attribute = 'any'
        #self.opr.par.nonAlignments = self.opr.par.leave_non_alignments_out
        self.opr.par.nonAlignments = False
        self.opr.par.maximum = -1
        self.opr.par.alignParser = xml.parsers.expat.ParserCreate()
        self.opr.par.alignParser.StartElementHandler = \
            self.opr.par.start_element
        self.opr.write_mode='normal'
        self.opr.par.sPar.wmode = 'normal'
        self.opr.par.sPar.pre = 'xml'
        self.opr.par.tPar.wmode = 'normal'
        self.opr.par.tPar.pre = 'xml'
        self.opr.par.sPar.annotations = False
        self.opr.par.tPar.annotations = False
        self.opr.change_moses_delimiter = '\t'
        self.fastopr.par.write_mode='normal'
        self.fastopr.par.slim=['all']
        self.fastopr.par.tlim=['all']
        self.fastopr.par.attribute = 'any'
        self.fastopr.par.nonAlignments = False
            #self.fastopr.par.leave_non_alignments_out
        self.fastopr.par.maximum = -1
        self.fastopr.par.alignParser = xml.parsers.expat.ParserCreate()
        self.fastopr.par.alignParser.StartElementHandler = \
            self.fastopr.par.start_element
        self.fastopr.par.sPar.wmode = 'normal'
        self.fastopr.par.sPar.pre = 'xml'
        self.fastopr.par.tPar.wmode = 'normal'
        self.fastopr.par.tPar.pre = 'xml'
        self.fastopr.par.sPar.annotations = False
        self.fastopr.par.tPar.annotations = False

    def test_ExhaustiveSentenceParser_initializing(self):
        self.assertEqual(len(self.opr.par.sPar.sentences), 29)
        self.assertEqual(len(self.opr.par.tPar.sentences), 68)

    def test_ExhaustiveSentenceParser_getSentence(self):
        self.assertEqual(self.opr.par.sPar.getSentence('s3.1')[0],
            '( Unofficial translation )')
        self.assertEqual(self.opr.par.tPar.getSentence('s3.1')[0],
            'Fru talman , ärade ledamöter av Sveriges riksdag !')

        self.assertEqual(self.opr.par.sPar.getSentence('s8.1')[0],
            ("The Government 's policy to combat unemployment will re"
            "st on four corner- stones :"))
        self.assertEqual(self.opr.par.tPar.getSentence('s8.1')[0],
            'Goda statsfinanser är grunden för alla politiska ambitioner .')

        self.assertEqual(self.opr.par.sPar.getSentence('s10.1')[0],
            'Sound public finances are the basis for all political a'
            'mbitions .')
        self.assertEqual(self.opr.par.tPar.getSentence('s10.1')[0],
            'Den andra hörnstenen är goda villkor för företag och fö'
            'retagande .')

    def test_ExhaustiveSentenceParser_readSentence_format(self):
        self.assertEqual(self.opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">( Unofficial translation )')
        self.assertEqual(self.opr.par.tPar.readSentence(['s3.1'])[0],
            '(trg)="s3.1">Fru talman , ärade ledamöter av Sveriges riksdag !')
        self.assertEqual(self.opr.par.sPar.readSentence(['s3.1', 's10.1'])[0],
            '(src)="s3.1">( Unofficial translation )\n'
            '(src)="s10.1">Sound public finances are the basis for all'
            ' political ambitions .')

    def test_ExhaustiveSentenceParser_readSentence_moses(self):
        self.opr.par.sPar.wmode = 'moses'
        self.assertEqual(self.opr.par.sPar.readSentence(['s3.1'])[0],
            '( Unofficial translation )')

    def test_ExhaustiveSentenceParser_readSentence_tmx(self):
        self.opr.par.sPar.wmode = 'tmx'
        self.assertEqual(self.opr.par.sPar.readSentence(['s3.1'])[0],
            '\t\t<tu>\n\t\t\t<tuv xml:lang="en"><seg>( Unofficial tr'
            'anslation )</seg></tuv>')
        self.opr.par.tPar.wmode = 'tmx'
        self.assertEqual(self.opr.par.tPar.readSentence(['s5.1', 's5.2'])[0],
            '\t\t\t<tuv xml:lang="sv"><seg>Fundamenten för ett gott '
            'samhälle undergrävs av dagens höga arbetslöshet . Såväl '
            'samhällsekonomi som moral och vilja försvagas .</seg></t'
            'uv>\n\t\t</tu>')

    def test_ExhaustiveSentenceParser_readSentence_empty(self):
        self.assertEqual(self.opr.par.sPar.readSentence([''])[0], '')

    def test_SentenceParser_readSentence_format(self):
        self.assertEqual(self.fastopr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">( Unofficial translation )')
        self.assertEqual(self.fastopr.par.tPar.readSentence(['s3.1'])[0],
            '(trg)="s3.1">Fru talman , ärade ledamöter av Sveriges riksdag !')
        self.assertEqual(self.fastopr.par.sPar.readSentence(
            ['s8.1', 's10.1'])[0],
            """(src)="s8.1">The Government 's policy to combat unemp"""
            """loyment will rest on four corner- stones :\n(src)="s10"""
            """.1">Sound public finances are the basis for all politi"""
            """cal ambitions .""")

    def test_SentenceParser_readSentence_moses(self):
        self.fastopr.par.sPar.wmode = 'moses'
        self.assertEqual(self.fastopr.par.sPar.readSentence(['s13.1'])[0],
            'Sweden is a good country for enterprise .')

    def test_SentenceParser_readSentence_tmx(self):
        self.fastopr.par.sPar.wmode = 'tmx'
        self.assertEqual(self.fastopr.par.sPar.readSentence(['s15.1'])[0],
            '\t\t<tu>\n\t\t\t<tuv xml:lang="en"><seg>The local and r'
            'egional role of economic policy will be emphasized .</se'
            'g></tuv>')
        self.fastopr.par.tPar.wmode = 'tmx'
        self.assertEqual(self.fastopr.par.tPar.readSentence(
            ['s11.1', 's11.2'])[0],
            '\t\t\t<tuv xml:lang="sv"><seg>Sverige är ett bra land f'
            'ör företagsamhet . Här finns en flexibel ekonomi , ett k'
            'onstruktivt samarbetsklimat och en kunnig och välutbilda'
            'd arbetskraft .</seg></tuv>\n\t\t</tu>')

    def test_SentenceParser_readSentence_empty(self):
        self.assertEqual(self.fastopr.par.sPar.readSentence([''])[0], '')

    def test_AlignmentParser_readPair_returns_1_if_tag_is_not_link_and_write_mode_is_links(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            write_mode='links', root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        opr.par.parseLine('<s>')
        ret = opr.par.readPair()
        self.assertEqual(ret, 1)
        opr.par.closeFiles()

    def test_AlignmentParser_readPair_returns_minus_1_if_tag_is_not_link_and_write_mode_id_not_links(self):
        self.opr.par.parseLine('<s>')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

    def test_AlignmentParser_readPair_returns_minus_1_if_number_of_sentences_is_outside_limit(self):
        self.opr.par.slim = ['0']
        self.opr.par.parseLine('<s>')
        self.opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

        self.opr.par.slim = ['1']
        self.opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        self.opr.par.tlim = ['3', '4']
        self.opr.par.parseLine('<link xtargets="s1;s1 s2" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

        self.opr.par.parseLine('<link xtargets="s1;s1 s2 s3" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        self.opr.par.parseLine('<link xtargets="s1;s1 s2 s3 s4" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        self.opr.par.parseLine('<link xtargets="s1 s2;s1 s2 s3 s4" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

        self.opr.par.parseLine('<link xtargets=";s1 s2 s3 s4" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

        self.opr.par.parseLine('<link xtargets="s1;s1 s2 s3 s4 s5" id="SL1"/>')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

    def test_AlignmentParser_readPair_returns_minus_1_if_attribute_values_is_not_over_threshold(self):
        self.opr.par.attribute = 'certainty'
        self.opr.par.threshold = 0.6
        self.opr.par.parseLine('<s>')
        self.opr.par.parseLine(
            '<link xtargets="s1;s1" id="SL1" certainty="0.5"/> ')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)
        self.opr.par.parseLine(
            '<link xtargets="s1;s1" id="SL1" certainty="0.7"/> ')
        ret = self.opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

    def test_AlignmentParser_readPair_returns_minus_1_if_nonAlignments_is_on_and_source_or_target_is_empty(self):
        self.opr.par.parseLine('<s>')
        self.opr.par.parseLine('<link xtargets="s1;" id="SL1"/> ')
        ret = self.opr.par.readPair()
        self.assertEqual(type(ret[0]), str)
        self.opr.par.nonAlignments = True
        self.opr.par.parseLine('<link xtargets="s1;" id="SL1"/> ')
        ret = self.opr.par.readPair()
        self.assertEqual(ret, -1)

    def test_AlignmentParser_readPair_returns_1_if_alignment_is_valid_and_write_mode_is_links(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            write_mode='links', root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/> ')
        ret = opr.par.readPair()
        self.assertEqual(ret, 1)

    def test_AlignmentParser_readPair_returns_minus_1_if_nonAlignments_is_on_and_source_or_target_is_empty_and_write_mode_is_links(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            leave_non_alignments_out=True, write_mode='links',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;" id="SL1"/> ')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

    def test_PairPrinter_printPair_normal(self):
        sPair = ('(src)="s4">Chapter 1 Mr. Sherlock Holmes',
                '(trg)="s4">Herra Sherlock Holmes .')
        self.assertEqual(self.opr.printPair(sPair),
            '(src)="s4">Chapter 1 Mr. Sherlock Holmes\n(trg)="s4">Herra '
            'Sherlock Holmes .\n================================')

    def test_PairPrinter_printPair_tmx(self):
        self.opr.write_mode = 'tmx'
        sPair = ('\t\t<tu>\n\t\t\t<tuv xml:lang="en"><seg>Chapter 1 Mr. '
                'Sherlock Holmes</seg></tuv>', '\t\t\t<tuv xml:lang="fi">'
                '<seg>Herra Sherlock Holmes .</seg></tuv>\n\t\t</tu>')
        self.assertEqual(self.opr.printPair(sPair),
            '\t\t<tu>\n\t\t\t<tuv xml:lang="en"><seg>Chapter 1 Mr. Sherlock'
            ' Holmes</seg></tuv>\n\t\t\t<tuv xml:lang="fi"><seg>Herra '
            'Sherlock Holmes .</seg></tuv>\n\t\t</tu>')

    def test_PairPrinter_printPair_moses(self):
        self.opr.write_mode = 'moses'
        sPair = ('Chapter 1 Mr. Sherlock Holmes', 'Herra Sherlock Holmes .')
        self.assertEqual(self.opr.printPair(sPair),
            """Chapter 1 Mr. Sherlock Holmes\tHerra Sherlock Holmes .""")

    def test_PairPrinter_printPair_moses_change_delimiter(self):
        self.opr.write_mode = 'moses'
        self.opr.change_moses_delimiter = '@'
        sPair = ('Chapter 1 Mr. Sherlock Holmes', 'Herra Sherlock Holmes .')
        self.assertEqual(self.opr.printPair(sPair),
            """Chapter 1 Mr. Sherlock Holmes@Herra Sherlock Holmes .""")

    def test_PairPrinter_printPair_links(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            root_directory=self.root_directory, write_mode='links')
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        sPair = '<link xtargets="s4;s4" id="SL4"/>'
        self.assertEqual(opr.printPair(sPair),
            '<link xtargets="s4;s4" id="SL4"/>')

    def test_PairPrinter_printPair_empty(self):
        sPair = ('(src)="3">Director PARK Jae-sik', '')
        self.assertEqual(self.opr.printPair(sPair),
            '(src)="3">Director PARK Jae-sik\n\n'
            '================================')

    def test_PairPrinter_writePair_normal(self):
        sPair = ('(src)="s4">Chapter 1 Mr. Sherlock Holmes',
                '(trg)="s4">Herra Sherlock Holmes .')
        self.assertEqual(self.opr.writePair(sPair),
            ('(src)="s4">Chapter 1 Mr. Sherlock Holmes\n(trg)="s4">Herra '
            'Sherlock Holmes .\n================================\n', ''))

    def test_PairPrinter_writePair_tmx(self):
        self.opr.write_mode = 'tmx'
        sPair = ('\t\t<tu>\n\t\t\t<tuv xml:lang="en"><seg>Chapter 1 Mr. '
                'Sherlock Holmes</seg></tuv>',
                '\t\t\t<tuv xml:lang="fi"><seg>Herra Sherlock Holmes .'
                '</seg></tuv>\n\t\t</tu>')
        self.assertEqual(self.opr.writePair(sPair),
            ('\t\t<tu>\n\t\t\t<tuv xml:lang="en"><seg>Chapter 1 Mr. Sherlock'
            ' Holmes</seg></tuv>\n\t\t\t<tuv xml:lang="fi"><seg>Herra '
            'Sherlock Holmes .</seg></tuv>\n\t\t</tu>\n', ''))

    def test_PairPrinter_writePair_moses(self):
        self.opr.write_mode = 'moses'
        self.opr.write = os.path.join(self.tempdir1, 'test_files',
            'test.src')
        sPair = ('Chapter 1 Mr. Sherlock Holmes', 'Herra Sherlock Holmes .')
        self.assertEqual(self.opr.writePair(sPair),
            ('Chapter 1 Mr. Sherlock Holmes\nHerra Sherlock Holmes .\n', ''))

    def test_PairPrinter_writePair_links(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            root_directory=self.root_directory, write_mode='links')
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        sPair = '<link xtargets="s4;s4" id="SL4"/>'
        self.assertEqual(opr.writePair(sPair),
            ('<link xtargets="s4;s4" id="SL4"/>\n', ''))

    def test_PairPrinter_writePair_empty(self):
        sPair = ('(src)="3">Director PARK Jae-sik', '')
        self.assertEqual(self.opr.writePair(sPair),
            ('(src)="3">Director PARK Jae-sik\n\n'
            '================================\n', ''))

    def test_switch_labels_when_languages_are_in_unalphabetical_order(self):
        opr = OpusRead(directory='RF', source='sv', target='en',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(trg)="s3.1">( Unofficial translation )')
        self.assertEqual(opr.par.tPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">Fru talman , ärade ledamöter av Sveriges riksdag !')

        opr.par.closeFiles()
        fastopr = OpusRead(directory='RF', source='sv', target='en',
            fast=True, root_directory=self.root_directory)
        fastopr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(fastopr.par.sPar.readSentence(['s3.1'])[0],
            '(trg)="s3.1">( Unofficial translation )')
        self.assertEqual(fastopr.par.tPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">Fru talman , ärade ledamöter av Sveriges riksdag !')
        fastopr.par.closeFiles()

    def test_ExhaustiveSentenceParser_readSentence_annotations(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            print_annotations=True, root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">(|(|( Unofficial|NNP|unofficial translatio'
            'n|NN|translation )|)|)')
        opr.par.closeFiles()
        opr = OpusRead(directory='RF', source='en', target='sv',
            print_annotations=True, change_annotation_delimiter='@',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">(@(@( Unofficial@NNP@unofficial translatio'
            'n@NN@translation )@)@)')
        opr.par.closeFiles()

    def test_ExhaustiveSentenceParser_readSentence_raw(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            preprocess='raw', root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">(Unofficial translation)')
        opr.par.closeFiles()

    def test_SentenceParser_readSentence_annotations(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            print_annotations=True, fast=True,
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">(|(|( Unofficial|NNP|unofficial translatio'
            'n|NN|translation )|)|)')
        opr.par.closeFiles()

    def test_SentenceParser_readSentence_annotations_change_delimiter(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            print_annotations=True, fast=True,
            change_annotation_delimiter='@',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">(@(@( Unofficial@NNP@unofficial translatio'
            'n@NN@translation )@)@)')
        opr.par.closeFiles()

    def test_SentenceParser_readSentence_raw(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            preprocess='raw', fast=True, root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sPar.readSentence(['s3.1'])[0],
            '(src)="s3.1">(Unofficial translation)')
        opr.par.closeFiles()

    def test_AlignmentParser_readPair_sentence_limits(self):
        opr = OpusRead(directory='RF', source='en', target='sv', tgt_range='0',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.closeFiles()

        opr = OpusRead(directory='RF', source='en', target='sv', tgt_range='1',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        opr.par.closeFiles()

        opr = OpusRead(directory='RF', source='en', target='sv',
            src_range='3-4', tgt_range='1', root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1 s2" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.parseLine('<link xtargets="s1 s2 s3;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        opr.par.parseLine('<link xtargets="s1 s2 s3 s4;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        opr.par.parseLine('<link xtargets="s1 s2 s3 s4;s1 s2" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.parseLine('<link xtargets="s1 s2 s3 s4;" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.parseLine('<link xtargets="s1 s2 s3 s4 s5;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.closeFiles()

    def test_AlignmentParser_readPair_sentence_limits_when_languages_in_unalphabetical_order(self):
        opr = OpusRead(directory='RF', source='sv', target='en', tgt_range='0',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.closeFiles()

        opr = OpusRead(directory='RF', source='sv', target='en', tgt_range='1',
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        opr.par.closeFiles()

        opr = OpusRead(directory='RF', source='sv', target='en',
            src_range='3-4', tgt_range='1', root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})

        opr.par.parseLine('<s>')
        opr.par.parseLine('<link xtargets="s1;s1 s2" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.parseLine('<link xtargets="s1;s1 s2 s3" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        opr.par.parseLine('<link xtargets="s1;s1 s2 s3 s4" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(type(ret[0]), str)

        opr.par.parseLine('<link xtargets="s1 s2;s1 s2 s3 s4" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.parseLine('<link xtargets=";s1 s2 s3 s4" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.parseLine('<link xtargets="s1;s1 s2 s3 s4 s5" id="SL1"/>')
        ret = opr.par.readPair()
        self.assertEqual(ret, -1)

        opr.par.closeFiles()

    def test_AlignmentParser_previous_document_is_closed_before_next_is_opened(self):
        printer = OpusRead(directory='RF', source='en', target='sv',
            root_directory=self.root_directory)
        printer.printPairs()
        self.assertEqual(True, True)

    def test_normal_xml_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n# en/1988.xml.gz\n'
                '# sv/1988.xml.gz\n\n'
                '================================\n(src)="s1.1">State'
                'ment of Government Policy by the Prime Minister , Mr'
                ' Ingvar Carlsson , at the Opening of the Swedish Parl'
                'iament on Tuesday , 4 October , 1988 .\n(trg)="s1.1"'
                '>REGERINGSFÖRKLARING .\n============================'
                '====\n')

    def test_normal_xml_write_verbose(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write=[os.path.join(
                self.tempdir1, 'test_files', 'test_result')],
            root_directory=self.root_directory, verbose=True)
        self.assertEqual(var,
            'Reading alignment file "{alignment}"\n'
            'Opening zip archive "{source}" ... Done\n'
            'Opening zip archive "{target}" ... Done\n'
            'Reading source file "RF/xml/en/1988.xml" and target file '
            '"RF/xml/sv/1988.xml"\nDone\n'.format(
                alignment=os.path.join(self.root_directory, 'RF', 'latest',
                    'xml', 'en-sv.xml.gz'),
                source=os.path.join(self.root_directory, 'RF', 'latest',
                    'xml', 'en.zip'),
                target=os.path.join(self.root_directory, 'RF', 'latest',
                    'xml', 'sv.zip')
                ))

    def test_normal_xml_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            fast=True, root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n# en/1988.xml.gz\n'
                '# sv/1988.xml.gz\n\n'
                '================================\n(src)="s1.1">State'
                'ment of Government Policy by the Prime Minister , Mr'
                ' Ingvar Carlsson , at the Opening of the Swedish Parl'
                'iament on Tuesday , 4 October , 1988 .\n(trg)="s1.1"'
                '>REGERINGSFÖRKLARING .\n============================'
                '====\n')

    def test_normal_xml_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n'
            '# sv/1988.xml.gz\n\n'
            '================================\n(src)="s1.1">State'
            'ment of Government Policy by the Prime Minister , Mr'
            ' Ingvar Carlsson , at the Opening of the Swedish Parl'
            'iament on Tuesday , 4 October , 1988 .\n(trg)="s1.1"'
            '>REGERINGSFÖRKLARING .\n============================'
            '====\n')

    def test_normal_xml_print_verbose(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, root_directory=self.root_directory, verbose=True)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n'
            '# sv/1988.xml.gz\n\n'
            '================================\n(src)="s1.1">State'
            'ment of Government Policy by the Prime Minister , Mr'
            ' Ingvar Carlsson , at the Opening of the Swedish Parl'
            'iament on Tuesday , 4 October , 1988 .\n(trg)="s1.1"'
            '>REGERINGSFÖRKLARING .\n============================'
            '====\n')

    def test_normal_xml_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, fast=True, root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n'
            '# sv/1988.xml.gz\n\n'
            '================================\n(src)="s1.1">State'
            'ment of Government Policy by the Prime Minister , Mr'
            ' Ingvar Carlsson , at the Opening of the Swedish Parl'
            'iament on Tuesday , 4 October , 1988 .\n(trg)="s1.1"'
            '>REGERINGSFÖRKLARING .\n============================'
            '====\n')

    def test_normal_raw_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            preprocess='raw', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n# en/1988.xml.gz\n'
                '# sv/1988.xml.gz\n\n'
                '================================\n(src)="s1.1">State'
                'ment of Government Policy by the Prime Minister, Mr'
                ' Ingvar Carlsson, at the Opening of the Swedish Parl'
                'iament on Tuesday, 4 October, 1988.\n(trg)="s1.1"'
                '>REGERINGSFÖRKLARING.\n============================'
                '====\n')

    def test_normal_raw_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            preprocess='raw', fast=True, root_directory=self.root_directory
            ).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n# en/1988.xml.gz\n'
                '# sv/1988.xml.gz\n\n'
                '================================\n(src)="s1.1">State'
                'ment of Government Policy by the Prime Minister, Mr'
                ' Ingvar Carlsson, at the Opening of the Swedish Parl'
                'iament on Tuesday, 4 October, 1988.\n(trg)="s1.1"'
                '>REGERINGSFÖRKLARING.\n============================'
                '====\n')

    def test_normal_raw_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, preprocess='raw', root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n'
            '# sv/1988.xml.gz\n\n'
            '================================\n(src)="s1.1">State'
            'ment of Government Policy by the Prime Minister, Mr'
            ' Ingvar Carlsson, at the Opening of the Swedish Parl'
            'iament on Tuesday, 4 October, 1988.\n(trg)="s1.1"'
            '>REGERINGSFÖRKLARING.\n============================'
            '====\n')

    def test_normal_raw_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, preprocess='raw', fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n'
            '# sv/1988.xml.gz\n\n'
            '================================\n(src)="s1.1">State'
            'ment of Government Policy by the Prime Minister, Mr'
            ' Ingvar Carlsson, at the Opening of the Swedish Parl'
            'iament on Tuesday, 4 October, 1988.\n(trg)="s1.1"'
            '>REGERINGSFÖRKLARING.\n============================'
            '====\n')

    def test_normal_raw_print_OpenSubtitles(self):
        var = pairPrinterToVariable(directory='OpenSubtitles', source='eo',
            target='tl', maximum=1, preprocess='raw',
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# eo/2009/1187043/6483790.xml.gz\n'
            '# tl/2009/1187043/6934998.xml.gz\n\n'
             '================================\n'
             '(src)="1">Ĉiuj nomoj, roluloj kaj eventoj reprezentitaj en ĉi '
             'tiu filmo estas fikciaj.\n\n'
             '================================\n')

    def test_normal_raw_print_OpenSubtitles_fast(self):
        var = pairPrinterToVariable(directory='OpenSubtitles', source='eo',
            target='tl', maximum=1, preprocess='raw', fast=True,
            root_directory= self.root_directory)
        self.assertEqual(var,
            '\n# eo/2009/1187043/6483790.xml.gz\n'
            '# tl/2009/1187043/6934998.xml.gz\n\n'
             '================================\n(src)="1">Ĉiuj nomoj, '
             'roluloj kaj eventoj reprezentitaj en ĉi tiu filmo estas '
             'fikciaj.\n\n================================\n')

    def test_normal_parsed_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
                '================================'
                '\n(src)="s1.1">Statement|NOUN|Number=Sing|statement '
                'of|ADP|of Government|NOUN|Number=Sing|government Pol'
                'icy|NOUN|Number=Sing|policy by|ADP|by the|DET|Defini'
                'te=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prim'
                'e Minister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|P'
                'ROPN|Number=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar '
                'Carlsson|PROPN|Number=Sing|Carlsson ,|PUNCT|, at|ADP'
                '|at the|DET|Definite=Def|PronType=Art|the Opening|NO'
                'UN|Number=Sing|opening of|ADP|of the|DET|Definite=De'
                'f|PronType=Art|the Swedish|ADJ|Degree=Pos|swedish Pa'
                'rliament|NOUN|Number=Sing|parliament on|ADP|on Tuesd'
                'ay|PROPN|Number=Sing|Tuesday ,|PUNCT|, 4|NUM|NumType'
                '=Card|4 October|PROPN|Number=Sing|October ,|PUNCT|, '
                '1988|NUM|NumType=Card|1988 .|PUNCT|.\n(trg)="s1.1">R'
                'EGERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender'
                '=Neut|Number=Sing|Regeringsförklaring .|PUNCT|.'
                '\n================================\n')

    def test_normal_parsed_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            fast=True, root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
                '================================'
                '\n(src)="s1.1">Statement|NOUN|Number=Sing|statement '
                'of|ADP|of Government|NOUN|Number=Sing|government Pol'
                'icy|NOUN|Number=Sing|policy by|ADP|by the|DET|Defini'
                'te=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prim'
                'e Minister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|P'
                'ROPN|Number=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar '
                'Carlsson|PROPN|Number=Sing|Carlsson ,|PUNCT|, at|ADP'
                '|at the|DET|Definite=Def|PronType=Art|the Opening|NO'
                'UN|Number=Sing|opening of|ADP|of the|DET|Definite=De'
                'f|PronType=Art|the Swedish|ADJ|Degree=Pos|swedish Pa'
                'rliament|NOUN|Number=Sing|parliament on|ADP|on Tuesd'
                'ay|PROPN|Number=Sing|Tuesday ,|PUNCT|, 4|NUM|NumType'
                '=Card|4 October|PROPN|Number=Sing|October ,|PUNCT|, '
                '1988|NUM|NumType=Card|1988 .|PUNCT|.\n(trg)="s1.1">R'
                'EGERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender'
                '=Neut|Number=Sing|Regeringsförklaring .|PUNCT|.'
                '\n================================\n')

    def test_normal_parsed_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
            '================================'
            '\n(src)="s1.1">Statement|NOUN|Number=Sing|statement '
            'of|ADP|of Government|NOUN|Number=Sing|government Pol'
            'icy|NOUN|Number=Sing|policy by|ADP|by the|DET|Defini'
            'te=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prim'
            'e Minister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|P'
            'ROPN|Number=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar '
            'Carlsson|PROPN|Number=Sing|Carlsson ,|PUNCT|, at|ADP'
            '|at the|DET|Definite=Def|PronType=Art|the Opening|NO'
            'UN|Number=Sing|opening of|ADP|of the|DET|Definite=De'
            'f|PronType=Art|the Swedish|ADJ|Degree=Pos|swedish Pa'
            'rliament|NOUN|Number=Sing|parliament on|ADP|on Tuesd'
            'ay|PROPN|Number=Sing|Tuesday ,|PUNCT|, 4|NUM|NumType'
            '=Card|4 October|PROPN|Number=Sing|October ,|PUNCT|, '
            '1988|NUM|NumType=Card|1988 .|PUNCT|.\n(trg)="s1.1">R'
            'EGERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender'
            '=Neut|Number=Sing|Regeringsförklaring .|PUNCT|.'
            '\n================================\n')

    def test_normal_parsed_print_unalphabetical(self):
        var = pairPrinterToVariable(directory='RF', source='sv', target='en',
            maximum=1, preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
            '================================'
            '\n(src)="s1.1">REGERINGSFÖRKLARING|NOUN|Case=Nom|Definit'
            'e=Ind|Gender=Neut|Number=Sing|Regeringsförklaring .|PUNC'
            'T|.\n(trg)="s1.1">Statement|NOUN|Number=Sing|statement '
            'of|ADP|of Government|NOUN|Number=Sing|government Pol'
            'icy|NOUN|Number=Sing|policy by|ADP|by the|DET|Defini'
            'te=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prim'
            'e Minister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|P'
            'ROPN|Number=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar '
            'Carlsson|PROPN|Number=Sing|Carlsson ,|PUNCT|, at|ADP'
            '|at the|DET|Definite=Def|PronType=Art|the Opening|NO'
            'UN|Number=Sing|opening of|ADP|of the|DET|Definite=De'
            'f|PronType=Art|the Swedish|ADJ|Degree=Pos|swedish Pa'
            'rliament|NOUN|Number=Sing|parliament on|ADP|on Tuesd'
            'ay|PROPN|Number=Sing|Tuesday ,|PUNCT|, 4|NUM|NumType'
            '=Card|4 October|PROPN|Number=Sing|October ,|PUNCT|, '
            '1988|NUM|NumType=Card|1988 .|PUNCT|.'
            '\n================================\n')

    def test_normal_parsed_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            fast=True, root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
            '================================'
            '\n(src)="s1.1">Statement|NOUN|Number=Sing|statement '
            'of|ADP|of Government|NOUN|Number=Sing|government Pol'
            'icy|NOUN|Number=Sing|policy by|ADP|by the|DET|Defini'
            'te=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prim'
            'e Minister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|P'
            'ROPN|Number=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar '
            'Carlsson|PROPN|Number=Sing|Carlsson ,|PUNCT|, at|ADP'
            '|at the|DET|Definite=Def|PronType=Art|the Opening|NO'
            'UN|Number=Sing|opening of|ADP|of the|DET|Definite=De'
            'f|PronType=Art|the Swedish|ADJ|Degree=Pos|swedish Pa'
            'rliament|NOUN|Number=Sing|parliament on|ADP|on Tuesd'
            'ay|PROPN|Number=Sing|Tuesday ,|PUNCT|, 4|NUM|NumType'
            '=Card|4 October|PROPN|Number=Sing|October ,|PUNCT|, '
            '1988|NUM|NumType=Card|1988 .|PUNCT|.\n(trg)="s1.1">R'
            'EGERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender'
            '=Neut|Number=Sing|Regeringsförklaring .|PUNCT|.'
            '\n================================\n')

    def test_normal_parsed_print_all_attributes(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, preprocess='parsed', print_annotations=True,
            source_annotations=['all_attrs'], target_annotations=['all_attrs'],
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
            '================================'
            '\n(src)="s1.1">Statement|root|Number=Sing|0|w1.1.1|state'
            'ment|NOUN|NOUN of|case|w1.1.4|w1.1.2|of|ADP|ADP Governme'
            'nt|compound|Number=Sing|w1.1.4|w1.1.3|government|NOUN|NO'
            'UN Policy|nmod|Number=Sing|w1.1.1|w1.1.4|policy|NOUN|NOU'
            'N by|case|w1.1.8|w1.1.5|by|ADP|ADP the|det|Definite=Def|'
            'PronType=Art|w1.1.8|w1.1.6|the|DET|DET Prime|compound|Nu'
            'mber=Sing|w1.1.8|w1.1.7|Prime|PROPN|PROPN Minister|nmod|'
            'Number=Sing|w1.1.1|w1.1.8|Minister|SpaceAfter=No|PROPN|P'
            'ROPN ,|punct|w1.1.8|w1.1.9|,|PUNCT|PUNCT Mr|compound|Num'
            'ber=Sing|w1.1.12|w1.1.10|Mr|PROPN|PROPN Ingvar|flat|Numb'
            'er=Sing|w1.1.10|w1.1.11|Ingvar|PROPN|PROPN Carlsson|flat'
            '|Number=Sing|w1.1.8|w1.1.12|Carlsson|SpaceAfter=No|PROPN'
            '|PROPN ,|punct|w1.1.1|w1.1.13|,|PUNCT|PUNCT at|case|w1.1'
            '.16|w1.1.14|at|ADP|ADP the|det|Definite=Def|PronType=Art'
            '|w1.1.16|w1.1.15|the|DET|DET Opening|nmod|Number=Sing|w1'
            '.1.1|w1.1.16|opening|NOUN|NOUN of|case|w1.1.20|w1.1.17|o'
            'f|ADP|ADP the|det|Definite=Def|PronType=Art|w1.1.20|w1.1'
            '.18|the|DET|DET Swedish|amod|Degree=Pos|w1.1.20|w1.1.19|'
            'swedish|ADJ|ADJ Parliament|nmod|Number=Sing|w1.1.16|w1.1'
            '.20|parliament|NOUN|NOUN on|case|w1.1.22|w1.1.21|on|ADP|'
            'ADP Tuesday|nmod|Number=Sing|w1.1.16|w1.1.22|Tuesday|Spa'
            'ceAfter=No|PROPN|PROPN ,|punct|w1.1.1|w1.1.23|,|PUNCT|PU'
            'NCT 4|nummod|NumType=Card|w1.1.25|w1.1.24|4|NUM|NUM Octo'
            'ber|appos|Number=Sing|w1.1.1|w1.1.25|October|SpaceAfter='
            'No|PROPN|PROPN ,|punct|w1.1.25|w1.1.26|,|PUNCT|PUNCT 198'
            '8|nummod|NumType=Card|w1.1.25|w1.1.27|1988|SpaceAfter=No'
            '|NUM|NUM .|punct|w1.1.1|w1.1.28|.|SpaceAfter=No|PUNCT|PU'
            'NCT\n(trg)="s1.1">REGERINGSFÖRKLARING|root|Case=Nom|Defini'
            'te=Ind|Gender=Neut|Number=Sing|0|w1.1.1|Regeringsförklar'
            'ing|SpaceAfter=No|NOUN|NOUN .|punct|w1.1.1|w1.1.2|.|Spac'
            'eAfter=No|PUNCT|PUNCT'
            '\n================================\n')

    def test_normal_parsed_print_all_attributes_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, preprocess='parsed', print_annotations=True,
            source_annotations=['all_attrs'], target_annotations=['all_attrs'],
            fast=True, root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n'
            '================================'
            '\n(src)="s1.1">Statement|root|Number=Sing|0|w1.1.1|state'
            'ment|NOUN|NOUN of|case|w1.1.4|w1.1.2|of|ADP|ADP Governme'
            'nt|compound|Number=Sing|w1.1.4|w1.1.3|government|NOUN|NO'
            'UN Policy|nmod|Number=Sing|w1.1.1|w1.1.4|policy|NOUN|NOU'
            'N by|case|w1.1.8|w1.1.5|by|ADP|ADP the|det|Definite=Def|'
            'PronType=Art|w1.1.8|w1.1.6|the|DET|DET Prime|compound|Nu'
            'mber=Sing|w1.1.8|w1.1.7|Prime|PROPN|PROPN Minister|nmod|'
            'Number=Sing|w1.1.1|w1.1.8|Minister|SpaceAfter=No|PROPN|P'
            'ROPN ,|punct|w1.1.8|w1.1.9|,|PUNCT|PUNCT Mr|compound|Num'
            'ber=Sing|w1.1.12|w1.1.10|Mr|PROPN|PROPN Ingvar|flat|Numb'
            'er=Sing|w1.1.10|w1.1.11|Ingvar|PROPN|PROPN Carlsson|flat'
            '|Number=Sing|w1.1.8|w1.1.12|Carlsson|SpaceAfter=No|PROPN'
            '|PROPN ,|punct|w1.1.1|w1.1.13|,|PUNCT|PUNCT at|case|w1.1'
            '.16|w1.1.14|at|ADP|ADP the|det|Definite=Def|PronType=Art'
            '|w1.1.16|w1.1.15|the|DET|DET Opening|nmod|Number=Sing|w1'
            '.1.1|w1.1.16|opening|NOUN|NOUN of|case|w1.1.20|w1.1.17|o'
            'f|ADP|ADP the|det|Definite=Def|PronType=Art|w1.1.20|w1.1'
            '.18|the|DET|DET Swedish|amod|Degree=Pos|w1.1.20|w1.1.19|'
            'swedish|ADJ|ADJ Parliament|nmod|Number=Sing|w1.1.16|w1.1'
            '.20|parliament|NOUN|NOUN on|case|w1.1.22|w1.1.21|on|ADP|'
            'ADP Tuesday|nmod|Number=Sing|w1.1.16|w1.1.22|Tuesday|Spa'
            'ceAfter=No|PROPN|PROPN ,|punct|w1.1.1|w1.1.23|,|PUNCT|PU'
            'NCT 4|nummod|NumType=Card|w1.1.25|w1.1.24|4|NUM|NUM Octo'
            'ber|appos|Number=Sing|w1.1.1|w1.1.25|October|SpaceAfter='
            'No|PROPN|PROPN ,|punct|w1.1.25|w1.1.26|,|PUNCT|PUNCT 198'
            '8|nummod|NumType=Card|w1.1.25|w1.1.27|1988|SpaceAfter=No'
            '|NUM|NUM .|punct|w1.1.1|w1.1.28|.|SpaceAfter=No|PUNCT|PU'
            'NCT\n(trg)="s1.1">REGERINGSFÖRKLARING|root|Case=Nom|Defini'
            'te=Ind|Gender=Neut|Number=Sing|0|w1.1.1|Regeringsförklar'
            'ing|SpaceAfter=No|NOUN|NOUN .|punct|w1.1.1|w1.1.2|.|Spac'
            'eAfter=No|PUNCT|PUNCT'
            '\n================================\n')

    def test_tmx_xml_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='tmx', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
                'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
                'n , at the Opening of the Swedish Parliament on Tues'
                'day , 4 October , 1988 .'
                '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
                'SFÖRKLARING .</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_xml_write_unalphabetical(self):
        OpusRead(directory='RF', source='sv', target='en', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='tmx', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="sv"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
                'SFÖRKLARING .</seg></tuv>\n\t\t\t<tuv xml:lang="en">'
                '<seg>Statement of Governm'
                'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
                'n , at the Opening of the Swedish Parliament on Tues'
                'day , 4 October , 1988 .'
                '</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_xml_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='tmx', fast=True, root_directory=self.root_directory
            ).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
                'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
                'n , at the Opening of the Swedish Parliament on Tues'
                'day , 4 October , 1988 .'
                '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
                'SFÖRKLARING .</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_xml_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
            'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
            'n , at the Opening of the Swedish Parliament on Tues'
            'day , 4 October , 1988 .'
            '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
            'SFÖRKLARING .</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_xml_print_verbose(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', root_directory=self.root_directory,
            verbose=True)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
            'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
            'n , at the Opening of the Swedish Parliament on Tues'
            'day , 4 October , 1988 .'
            '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
            'SFÖRKLARING .</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_xml_print_unalphabetical(self):
        var = pairPrinterToVariable(directory='RF', source='sv', target='en',
            maximum=1, write_mode='tmx', root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="sv"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
            'SFÖRKLARING .</seg></tuv>\n\t\t\t<tuv xml:lang="en"><seg'
            '>Statement of Governm'
            'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
            'n , at the Opening of the Swedish Parliament on Tues'
            'day , 4 October , 1988 .'
            '</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_xml_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
            'ent Policy by the Prime Minister , Mr Ingvar Carlsso'
            'n , at the Opening of the Swedish Parliament on Tues'
            'day , 4 October , 1988 .'
            '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
            'SFÖRKLARING .</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_raw_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write_mode='tmx',
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            preprocess='raw', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
                'ent Policy by the Prime Minister, Mr Ingvar Carlsso'
                'n, at the Opening of the Swedish Parliament on Tues'
                'day, 4 October, 1988.'
                '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
                'SFÖRKLARING.</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_raw_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='tmx', preprocess='raw', fast=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
                'ent Policy by the Prime Minister, Mr Ingvar Carlsso'
                'n, at the Opening of the Swedish Parliament on Tues'
                'day, 4 October, 1988.'
                '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
                'SFÖRKLARING.</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_raw_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', preprocess='raw',
            root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
            'ent Policy by the Prime Minister, Mr Ingvar Carlsso'
            'n, at the Opening of the Swedish Parliament on Tues'
            'day, 4 October, 1988.'
            '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
            'SFÖRKLARING.</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_raw_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', preprocess='raw', fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement of Governm'
            'ent Policy by the Prime Minister, Mr Ingvar Carlsso'
            'n, at the Opening of the Swedish Parliament on Tues'
            'day, 4 October, 1988.'
            '</seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERING'
            'SFÖRKLARING.</seg></tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_parsed_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='tmx', preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="en"><seg>Statement|NOUN|Numbe'
                'r=Sing|statement of|ADP|of Government|NOUN|Number=Si'
                'ng|government Policy|NOUN|Number=Sing|policy by|ADP|'
                'by the|DET|Definite=Def|PronType=Art|the Prime|PROPN'
                '|Number=Sing|Prime Minister|PROPN|Number=Sing|Minist'
                'er ,|PUNCT|, Mr|PROPN|Number=Sing|Mr Ingvar|PROPN|Nu'
                'mber=Sing|Ingvar Carlsson|PROPN|Number=Sing|Carlsson '
                ',|PUNCT|, at|ADP|at the|DET|Definite=Def|PronType=Ar'
                't|the Opening|NOUN|Number=Sing|opening of|ADP|of the'
                '|DET|Definite=Def|PronType=Art|the Swedish|ADJ|Degre'
                'e=Pos|swedish Parliament|NOUN|Number=Sing|parliament '
                'on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PUNCT|'
                ', 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
                'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.<'
                '/seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERINGS'
                'FÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Neut|Nu'
                'mber=Sing|Regeringsförklaring .|PUNCT|.</seg></tuv>'
                '\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_parsed_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_results')],
            write_mode='tmx', preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'], fast=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
                '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
                '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
                '\n\t\t\t<tuv xml:lang="en"><seg>Statement|NOUN|Numbe'
                'r=Sing|statement of|ADP|of Government|NOUN|Number=Si'
                'ng|government Policy|NOUN|Number=Sing|policy by|ADP|'
                'by the|DET|Definite=Def|PronType=Art|the Prime|PROPN'
                '|Number=Sing|Prime Minister|PROPN|Number=Sing|Minist'
                'er ,|PUNCT|, Mr|PROPN|Number=Sing|Mr Ingvar|PROPN|Nu'
                'mber=Sing|Ingvar Carlsson|PROPN|Number=Sing|Carlsson '
                ',|PUNCT|, at|ADP|at the|DET|Definite=Def|PronType=Ar'
                't|the Opening|NOUN|Number=Sing|opening of|ADP|of the'
                '|DET|Definite=Def|PronType=Art|the Swedish|ADJ|Degre'
                'e=Pos|swedish Parliament|NOUN|Number=Sing|parliament '
                'on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PUNCT|'
                ', 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
                'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.<'
                '/seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERINGS'
                'FÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Neut|Nu'
                'mber=Sing|Regeringsförklaring .|PUNCT|.</seg></tuv>'
                '\n\t\t</tu>\n\t</body>\n</tmx>')

    def test_tmx_parsed_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', preprocess='parsed',
            print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement|NOUN|Numbe'
            'r=Sing|statement of|ADP|of Government|NOUN|Number=Si'
            'ng|government Policy|NOUN|Number=Sing|policy by|ADP|'
            'by the|DET|Definite=Def|PronType=Art|the Prime|PROPN'
            '|Number=Sing|Prime Minister|PROPN|Number=Sing|Minist'
            'er ,|PUNCT|, Mr|PROPN|Number=Sing|Mr Ingvar|PROPN|Nu'
            'mber=Sing|Ingvar Carlsson|PROPN|Number=Sing|Carlsson '
            ',|PUNCT|, at|ADP|at the|DET|Definite=Def|PronType=Ar'
            't|the Opening|NOUN|Number=Sing|opening of|ADP|of the'
            '|DET|Definite=Def|PronType=Art|the Swedish|ADJ|Degre'
            'e=Pos|swedish Parliament|NOUN|Number=Sing|parliament '
            'on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PUNCT|'
            ', 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
            'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.<'
            '/seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERINGS'
            'FÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Neut|Nu'
            'mber=Sing|Regeringsförklaring .|PUNCT|.</seg></tuv>'
            '\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_tmx_parsed_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='tmx', preprocess='parsed',
            print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'], fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<tmx version="1.4.">'
            '\n<header srclang="en"\n\tadminlang="en"\n\tsegtype='
            '"sentence"\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Statement|NOUN|Numbe'
            'r=Sing|statement of|ADP|of Government|NOUN|Number=Si'
            'ng|government Policy|NOUN|Number=Sing|policy by|ADP|'
            'by the|DET|Definite=Def|PronType=Art|the Prime|PROPN'
            '|Number=Sing|Prime Minister|PROPN|Number=Sing|Minist'
            'er ,|PUNCT|, Mr|PROPN|Number=Sing|Mr Ingvar|PROPN|Nu'
            'mber=Sing|Ingvar Carlsson|PROPN|Number=Sing|Carlsson '
            ',|PUNCT|, at|ADP|at the|DET|Definite=Def|PronType=Ar'
            't|the Opening|NOUN|Number=Sing|opening of|ADP|of the'
            '|DET|Definite=Def|PronType=Art|the Swedish|ADJ|Degre'
            'e=Pos|swedish Parliament|NOUN|Number=Sing|parliament '
            'on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PUNCT|'
            ', 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
            'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.<'
            '/seg></tuv>\n\t\t\t<tuv xml:lang="sv"><seg>REGERINGS'
            'FÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Neut|Nu'
            'mber=Sing|Regeringsförklaring .|PUNCT|.</seg></tuv>'
            '\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_moses_xml_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
            'Statement of Government Policy by the Prime Minister , '
            'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
            'ament on Tuesday , 4 October , 1988 .\n')
        with open(os.path.join(self.tempdir1, 'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(), 'REGERINGSFÖRKLARING .\n')

    def test_moses_xml_write_unalphabetical(self):
        OpusRead(directory='RF', source='sv', target='en', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(),
            'Statement of Government Policy by the Prime Minister , '
            'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
            'ament on Tuesday , 4 October , 1988 .\n')
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(), 'REGERINGSFÖRKLARING .\n')

    def test_moses_xml_write_with_file_names(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', print_file_names=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
            '\n<fromDoc>en/1988.xml.gz</fromDoc>\n\nStatement of Gover'
            'nment Policy by the Prime Minister , '
            'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
            'ament on Tuesday , 4 October , 1988 .\n')
        with open(os.path.join(self.tempdir1, 'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(),
            '\n<toDoc>sv/1988.xml.gz</toDoc>\n\nREGERINGSFÖRKLARING .\n')

    def test_moses_xml_write_single_file(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src')],
            write_mode='moses', root_directory=self.root_directory
            ).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
                'Statement of Government Policy by the Prime Minister , '
                'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
                'ament on Tuesday , 4 October , 1988 .\tREGERINGSFÖRK'
                'LARING .\n')

    def test_moses_xml_write_single_file_unalphabetical(self):
        OpusRead(directory='RF', source='sv', target='en', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src')],
            write_mode='moses', root_directory=self.root_directory
            ).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
                'REGERINGSFÖRKLARING .\tStatement of Government Poli'
                'cy by the Prime Minister , Mr Ingvar Carlsson , at t'
                'he Opening of the Swedish Parliament on Tuesday , 4 '
                'October , 1988 .\n')

    def test_moses_xml_write_single_file_with_file_names(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src')],
            write_mode='moses', print_file_names=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n<fromDoc>en/1988.xml.gz</fromDoc>\n<toDoc>sv/1988'
                '.xml.gz</toDoc>\n\nStatement of Government Policy by'
                ' the Prime Minister , Mr Ingvar Carlsson , at the Ope'
                'ning of the Swedish Parliament on Tuesday , 4 Octobe'
                'r , 1988 .\tREGERINGSFÖRKLARING .\n')

    def test_moses_xml_write_single_file_with_file_names_unalphabetical(self):
        OpusRead(directory='RF', source='sv', target='en', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src')],
            write_mode='moses', print_file_names=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1,
            'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
                '\n<fromDoc>en/1988.xml.gz</fromDoc>\n<toDoc>sv/1988'
                '.xml.gz</toDoc>\n\nREGERINGSFÖRKLARING .\tStatement '
                'of Government Policy by the Prime Minister , Mr Ingv'
                'ar Carlsson , at the Opening of the Swedish Parliame'
                'nt on Tuesday , 4 October , 1988 .\n')

    def test_moses_xml_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', fast=True, root_directory=self.root_directory
            ).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
            'Statement of Government Policy by the Prime Minister , '
            'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
            'ament on Tuesday , 4 October , 1988 .\n')
        with open(os.path.join(self.tempdir1, 'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(), 'REGERINGSFÖRKLARING .\n')

    def test_moses_xml_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', root_directory=self.root_directory)
        self.assertEqual(var,
            'Statement of Government Policy by the Prime Minister , '
            'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
            'ament on Tuesday , 4 October , 1988 .\t'
            'REGERINGSFÖRKLARING .\n')

    def test_moses_xml_print_unalphabetical(self):
        var = pairPrinterToVariable(directory='RF', source='sv', target='en',
            maximum=1, write_mode='moses', root_directory=self.root_directory)
        self.assertEqual(var,
            'REGERINGSFÖRKLARING .\tStatement of Government Policy b'
            'y the Prime Minister , Mr Ingvar Carlsson , at the Openi'
            'ng of the Swedish Parliament on Tuesday , 4 October , 1988 .\n')

    def test_moses_xml_print_with_file_names(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', print_file_names=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n<fromDoc>en/1988.xml.gz</fromDoc>\n<toDoc>sv/1988'
            '.xml.gz</toDoc>\n\nStatement of Government Policy by'
            ' the Prime Minister , Mr Ingvar Carlsson , at the Ope'
            'ning of the Swedish Parliament on Tuesday , 4 Octobe'
            'r , 1988 .\tREGERINGSFÖRKLARING .\n')

    def test_moses_xml_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            'Statement of Government Policy by the Prime Minister , '
            'Mr Ingvar Carlsson , at the Opening of the Swedish Parli'
            'ament on Tuesday , 4 October , 1988 .\t'
            'REGERINGSFÖRKLARING .\n')

    def test_moses_raw_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write_mode='moses',
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            preprocess='raw', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1,
                'test_files', 'test.src'), 'r') as f:
            self.assertEqual(f.read(),
            'Statement of Government Policy by the Prime Minister, '
            'Mr Ingvar Carlsson, at the Opening of the Swedish Parli'
            'ament on Tuesday, 4 October, 1988.\n')
        with open(os.path.join(self.tempdir1,
                'test_files', 'test.trg'), 'r') as f:
            self.assertEqual(f.read(), 'REGERINGSFÖRKLARING.\n')

    def test_moses_raw_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', preprocess='raw', fast=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(),
            'Statement of Government Policy by the Prime Minister, '
            'Mr Ingvar Carlsson, at the Opening of the Swedish Parli'
            'ament on Tuesday, 4 October, 1988.\n')
        with open(os.path.join(self.tempdir1, 'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(), 'REGERINGSFÖRKLARING.\n')

    def test_moses_raw_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', preprocess='raw',
            root_directory=self.root_directory)
        self.assertEqual(var,
            'Statement of Government Policy by the Prime Minister, '
            'Mr Ingvar Carlsson, at the Opening of the Swedish Parli'
            'ament on Tuesday, 4 October, 1988.\t'
            'REGERINGSFÖRKLARING.\n')

    def test_moses_raw_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', preprocess='raw', fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            'Statement of Government Policy by the Prime Minister, '
            'Mr Ingvar Carlsson, at the Opening of the Swedish Parli'
            'ament on Tuesday, 4 October, 1988.\t'
            'REGERINGSFÖRKLARING.\n')

    def test_moses_parsed_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(), 'Statement|NOUN|Number=Sing|st'
            'atement of|ADP|of Government|NOUN|Number=Sing|government'
            ' Policy|NOUN|Number=Sing|policy by|ADP|by the|DET|Definit'
            'e=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prime Min'
            'ister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|PROPN|Numb'
            'er=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar Carlsson|PROP'
            'N|Number=Sing|Carlsson ,|PUNCT|, at|ADP|at the|DET|Defin'
            'ite=Def|PronType=Art|the Opening|NOUN|Number=Sing|openin'
            'g of|ADP|of the|DET|Definite=Def|PronType=Art|the Swedis'
            'h|ADJ|Degree=Pos|swedish Parliament|NOUN|Number=Sing|par'
            'liament on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PU'
            'NCT|, 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
            'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.\n')
        with open(os.path.join(self.tempdir1,
            'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(),
            'REGERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Ne'
            'ut|Number=Sing|Regeringsförklaring .|PUNCT|.\n')

    def test_moses_parsed_write_fast(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test.src'),
                os.path.join(self.tempdir1, 'test_files', 'test.trg')],
            write_mode='moses', preprocess='parsed', print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'], fast=True,
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test.src'),
                'r') as f:
            self.assertEqual(f.read(), 'Statement|NOUN|Number=Sing|st'
            'atement of|ADP|of Government|NOUN|Number=Sing|government'
            ' Policy|NOUN|Number=Sing|policy by|ADP|by the|DET|Definit'
            'e=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prime Min'
            'ister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|PROPN|Numb'
            'er=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar Carlsson|PROP'
            'N|Number=Sing|Carlsson ,|PUNCT|, at|ADP|at the|DET|Defin'
            'ite=Def|PronType=Art|the Opening|NOUN|Number=Sing|openin'
            'g of|ADP|of the|DET|Definite=Def|PronType=Art|the Swedis'
            'h|ADJ|Degree=Pos|swedish Parliament|NOUN|Number=Sing|par'
            'liament on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PU'
            'NCT|, 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
            'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.\n')
        with open(os.path.join(self.tempdir1, 'test_files', 'test.trg'),
                'r') as f:
            self.assertEqual(f.read(),
            'REGERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Ne'
            'ut|Number=Sing|Regeringsförklaring .|PUNCT|.\n')

    def test_moses_parsed_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', preprocess='parsed',
            print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'],
            root_directory=self.root_directory)
        self.assertEqual(var,
            'Statement|NOUN|Number=Sing|st'
            'atement of|ADP|of Government|NOUN|Number=Sing|government'
            ' Policy|NOUN|Number=Sing|policy by|ADP|by the|DET|Definit'
            'e=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prime Min'
            'ister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|PROPN|Numb'
            'er=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar Carlsson|PROP'
            'N|Number=Sing|Carlsson ,|PUNCT|, at|ADP|at the|DET|Defin'
            'ite=Def|PronType=Art|the Opening|NOUN|Number=Sing|openin'
            'g of|ADP|of the|DET|Definite=Def|PronType=Art|the Swedis'
            'h|ADJ|Degree=Pos|swedish Parliament|NOUN|Number=Sing|par'
            'liament on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PU'
            'NCT|, 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
            'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.\tREG'
            'ERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Ne'
            'ut|Number=Sing|Regeringsförklaring .|PUNCT|.\n')

    def test_moses_parsed_print_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='moses', preprocess='parsed',
            print_annotations=True,
            source_annotations=['upos', 'feats', 'lemma'],
            target_annotations=['upos', 'feats', 'lemma'], fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            'Statement|NOUN|Number=Sing|st'
            'atement of|ADP|of Government|NOUN|Number=Sing|government'
            ' Policy|NOUN|Number=Sing|policy by|ADP|by the|DET|Definit'
            'e=Def|PronType=Art|the Prime|PROPN|Number=Sing|Prime Min'
            'ister|PROPN|Number=Sing|Minister ,|PUNCT|, Mr|PROPN|Numb'
            'er=Sing|Mr Ingvar|PROPN|Number=Sing|Ingvar Carlsson|PROP'
            'N|Number=Sing|Carlsson ,|PUNCT|, at|ADP|at the|DET|Defin'
            'ite=Def|PronType=Art|the Opening|NOUN|Number=Sing|openin'
            'g of|ADP|of the|DET|Definite=Def|PronType=Art|the Swedis'
            'h|ADJ|Degree=Pos|swedish Parliament|NOUN|Number=Sing|par'
            'liament on|ADP|on Tuesday|PROPN|Number=Sing|Tuesday ,|PU'
            'NCT|, 4|NUM|NumType=Card|4 October|PROPN|Number=Sing|Oct'
            'ober ,|PUNCT|, 1988|NUM|NumType=Card|1988 .|PUNCT|.\tREG'
            'ERINGSFÖRKLARING|NOUN|Case=Nom|Definite=Ind|Gender=Ne'
            'ut|Number=Sing|Regeringsförklaring .|PUNCT|.\n')

    def test_links_write(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='links', root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<!DOCTYPE cesAlign PUBLIC "-//CES//DTD'
                ' XML cesAlign//EN" "">\n<cesAlign version="1.0">\n '
                '<linkGrp targType="s" toDoc="sv/1988.xml.gz"'
                ' fromDoc="en/1988.xml.gz">\n'
                '<link certainty="-0.0636364" xtargets="s1.1;s1.1" id="SL1"'
                ' />\n </linkGrp>\n</cesAlign>')

    def test_links_write_unalphabetical(self):
        OpusRead(directory='RF', source='sv', target='en',
            write=[os.path.join(self.tempdir1, 'test_files', 'test_result')],
            write_mode='links', src_range='1', tgt_range='2',
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'test_result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<!DOCTYPE cesAlign PUBLIC "-//CES//DTD'
                ' XML cesAlign//EN" "">\n<cesAlign version="1.0">\n '
                '<linkGrp targType="s" toDoc="sv/1988.xml.gz"'
                ' fromDoc="en/1988.xml.gz">\n'
                '<link certainty="0.188136" xtargets="s4.4 s4.5;s4.4" '
                'id="SL10" />\n </linkGrp>\n <linkGrp targType="s" '
                'toDoc="sv/1996.xml.gz" fromDoc="en/1996.xml.gz">\n '
                '</linkGrp>\n</cesAlign>\n')

    def test_links_print(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, write_mode='links', root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE cesAlign PUBLIC "-//CES//DTD'
            ' XML cesAlign//EN" "">\n<cesAlign version="1.0">\n '
            '<linkGrp targType="s" toDoc="sv/1988.xml.gz"'
            ' fromDoc="en/1988.xml.gz">\n'
            '<link certainty="-0.0636364" xtargets="s1.1;s1.1" id="SL1"'
            ' />\n </linkGrp>\n</cesAlign>\n')

    def test_links_print_unalphabetical(self):
        var = pairPrinterToVariable(directory='RF', source='sv', target='en',
            write_mode='links', src_range='1', tgt_range='2',
            root_directory=self.root_directory)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE cesAlign PUBLIC "-//CES//DTD'
            ' XML cesAlign//EN" "">\n<cesAlign version="1.0">\n '
            '<linkGrp targType="s" toDoc="sv/1988.xml.gz"'
            ' fromDoc="en/1988.xml.gz">\n'
            '<link certainty="0.188136" xtargets="s4.4 s4.5;s4.4" id="SL10"'
            ' />\n </linkGrp>\n <linkGrp targType="s" toDoc="sv/1996.xml.gz"'
            ' fromDoc="en/1996.xml.gz">\n </linkGrp>\n</cesAlign>\n')

    def test_iteration_stops_at_the_end_of_the_document_even_if_max_is_not_filled(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            src_range='2', tgt_range='1', maximum=5,
            root_directory=self.root_directory)
        self.assertEqual(var,
            """\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n=============="""
            """==================\n(src)="s4.4">The army will be reor"""
            """ganized with the aim of making it more effective .\n("""
            """src)="s4.5">It is the Government 's intention to seek """
            """broad solutions in issues that are of importance for o"""
            """ur national security .\n(trg)="s4.4">Det är regeringe"""
            """ns föresats att söka breda lösningar i frågor som är a"""
            """v betydelse för vår nationella säkerhet .\n=========="""
            """======================\n\n# en/1996.xml.gz\n# sv/1996."""
            """xml.gz\n\n================================\n""")


    def test_use_given_sentence_alignment_file(self):
        OpusRead(directory='Books', source='eo', target='pt', src_range='2',
            tgt_range='2', maximum=1, write_mode='links',
            write=[os.path.join(self.tempdir1, 'test_files', 'testlinks')],
            root_directory=self.root_directory).printPairs()
        var = pairPrinterToVariable(
            directory='Books', source='eo', target='pt',
            alignment_file=os.path.join(self.tempdir1, 'test_files',
                'testlinks'),
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# eo/Carroll_Lewis-Alice_in_wonderland.xml.gz\n'
            '# pt/Carroll_Lewis-Alice_in_wonderland.xml.gz\n\n======='
            '=========================\n'
            '(src)="s7.1">( Kiam poste ŝi pripensadis la aferon , ŝaj'
            'nis al ŝi , ke tio estis efektive mirinda , sed en la mo'
            'mento mem , ĝi ŝajnis al ŝi tute ordinara . )\n(src)="s7'
            '.2">Tamen , kiam la kuniklo el sia veŝta poŝo eligis poŝ'
            'horloĝon , kaj , rigardinte ĝin , tuj plirapidis , Alici'
            'o eksaltis sur la piedojn , ĉar subite frapis ŝin la ide'
            'o , ke neniam antaŭe ŝi vidis kuniklon kiu havas veŝtpoŝ'
            'on kaj poŝhorloĝon .\n(trg)="s7.1">Oh , céus !\n(trg)='
            '"s7.2">Irei me atrasar ! " ( quando refletiu sobre isso '
            'depois , ocorreu-lhe que deveria ter reparado nisso , ma'
            's à hora tudo lhe pareceu bastante natural ) ; mas quand'
            'o o Coelho efetivamente tirou um relógio do bolso do col'
            'ete e olhou para ele , se apressando , Alice pôs-se de p'
            'é porque lhe relampagueou pela cabeça que nunca tivera v'
            'isto antes um coelho nem com um bolso de colete , nem co'
            'm um relógio para tirar dele e , ardendo de curiosidade '
            ', correu através do campo atrás dele e felizmente chegou '
            'bem a tempo de o ver pular para dentro de uma grande toc'
            'a de coelho debaixo da cerca .\n========================'
            '========\n')

    def test_use_given_sentence_alignment_file_with_lingGrp_end_tag_on_the_same_line_as_link_tag(self):
        OpusRead(directory='RF', source='en', target='sv', src_range='2',
            tgt_range='1', write_mode='links',
            write=[os.path.join(self.tempdir1, 'test_files', 'testlinks')],
            root_directory=self.root_directory).printPairs()
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            alignment_file=os.path.join(self.tempdir1, 'test_files',
                'testlinks'), root_directory=self.root_directory)
        self.assertEqual(var,
            """\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n=============="""
            """==================\n(src)="s4.4">The army will be reor"""
            """ganized with the aim of making it more effective .\n("""
            """src)="s4.5">It is the Government 's intention to seek """
            """broad solutions in issues that are of importance for o"""
            """ur national security .\n(trg)="s4.4">Det är regeringe"""
            """ns föresats att söka breda lösningar i frågor som är a"""
            """v betydelse för vår nationella säkerhet .\n=========="""
            """======================\n\n# en/1996.xml.gz\n# sv/1996."""
            """xml.gz\n\n================================\n""")

    def test_use_given_sentence_alignment_file_and_print_links(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write_mode='links',
            write=[os.path.join(self.tempdir1, 'test_files', 'testlinks')],
            root_directory=self.root_directory).printPairs()
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            write_mode='links',
            alignment_file=os.path.join(self.tempdir1, 'test_files',
                'testlinks'),
            root_directory=self.root_directory)
        self.assertEqual(var, '<?xml version="1.0" encoding="utf-8"?>'
        '\n<!DOCTYPE cesAlign PUBLIC "-//CES//DTD XML cesAlign//EN" "'
        '">\n<cesAlign version="1.0">\n <linkGrp targType="s" toDoc="'
        'sv/1988.xml.gz" fromDoc="en/1988.xml.gz">\n<link certainty="'
        '-0.0636364" xtargets="s1.1;s1.1" id="SL1" />\n </linkGrp>\n<'
        '/cesAlign>\n')

    def test_use_given_sentence_alignment_file_and_write_links(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write_mode='links',
            write=[os.path.join(self.tempdir1, 'test_files', 'testlinks')],
            root_directory=self.root_directory).printPairs()
        OpusRead(directory='RF', source='en', target='sv', write_mode='links',
            alignment_file=os.path.join(self.tempdir1, 'test_files',
                'testlinks'),
            write=[os.path.join(self.tempdir1, 'test_files', 'testresult')],
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'testresult'),
                'r') as f:
            self.assertEqual(f.read(), '<?xml version="1.0" encoding="utf-8"?>'
            '\n<!DOCTYPE cesAlign PUBLIC "-//CES//DTD XML cesAlign//EN" "'
            '">\n<cesAlign version="1.0">\n <linkGrp targType="s" toDoc="'
            'sv/1988.xml.gz" fromDoc="en/1988.xml.gz">\n<link certainty="'
            '-0.0636364" xtargets="s1.1;s1.1" id="SL1" />\n </linkGrp>\n<'
            '/cesAlign>\n')

    def test_use_given_sentence_alignment_file_and_print_links_Books(self):
        OpusRead(directory='Books', source='eo', target='pt', maximum=1,
            write_mode='links',
            write=[os.path.join(self.tempdir1, 'test_files', 'testlinks')],
            root_directory=self.root_directory).printPairs()
        var = pairPrinterToVariable(directory='Books', source='eo',
            target='pt', write_mode='links',
            alignment_file=os.path.join(self.tempdir1, 'test_files',
                'testlinks'),
            root_directory=self.root_directory)
        self.assertEqual(var, '<?xml version="1.0" encoding="utf-8"?>'
        '\n<!DOCTYPE cesAlign PUBLIC "-//CES//DTD XML cesAlign//EN" "'
        '">\n<cesAlign version="1.0">\n<linkGrp targType="s" fromDoc='
        '"eo/Carroll_Lewis-Alice_in_wonderland.xml.gz" toDoc="pt/Carr'
        'oll_Lewis-Alice_in_wonderland.xml.gz" >\n<link xtargets="s1;'
        's1" id="SL1"/>\n </linkGrp>\n</cesAlign>\n')

    def test_use_given_sentence_alignment_file_and_write_links_Books(self):
        OpusRead(directory='Books', source='eo', target='pt', maximum=1,
            write_mode='links',
            write=[os.path.join(self.tempdir1, 'test_files', 'testlinks')],
            root_directory=self.root_directory).printPairs()
        OpusRead(directory='Books', source='eo', target='pt',
            write_mode='links', alignment_file=os.path.join(self.tempdir1,
                'test_files', 'testlinks'),
            write=[os.path.join(self.tempdir1, 'test_files', 'testresult')],
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'testresult'),
                'r') as f:
            self.assertEqual(f.read(), '<?xml version="1.0" encoding="utf-8"?>'
            '\n<!DOCTYPE cesAlign PUBLIC "-//CES//DTD XML cesAlign//EN" "'
            '">\n<cesAlign version="1.0">\n<linkGrp targType="s" from'
            'Doc="eo/Carroll_Lewis-Alice_in_wonderland.xml.gz" toDoc='
            '"pt/Carroll_Lewis-Alice_in_wonderland.xml.gz" >\n<link x'
            'targets="s1;s1" id="SL1"/>\n </linkGrp>\n</cesAlign>\n')

    def test_checks_first_whether_documents_are_in_path(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp>+\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s>\n </body>\n</text>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')

        var = pairPrinterToVariable(directory='Books', source='en',
            target='fi', alignment_file=os.path.join(self.tempdir1,
                'test_files', 'testlinks'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# test_files/test_en\n# test_files/test_fi\n\n'
            '================================\n(src)="s1">test_en1 test_en2\n'
            '(trg)="s1">test_fi1 test_fi2'
            '\n================================\n')

    def test_open_documents_from_specifed_zips(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp>+\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_en.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                arcname=os.path.join('test_files', 'test_en'))
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_fi.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                arcname=os.path.join('test_files', 'test_fi'))

        var = pairPrinterToVariable(directory='Books', source='en',
            target='fi', alignment_file=os.path.join(self.tempdir1,
                'test_files', 'testlinks'),
            source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
            target_zip = os.path.join(self.tempdir1, 'test_fi.zip'))
        self.assertEqual(var,
            '\n# test_files/test_en\n# test_files/test_fi\n\n'
            '================================\n(src)="s1">test_en1 test_en2\n'
            '(trg)="s1">test_fi1 test_fi2'
            '\n================================\n')

    def test_try_to_open_wrongly_named_docs_from_specifed_source_zip(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp>+\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_en.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                arcname=os.path.join('test_files', 'test_un'))
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_fi.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                arcname=os.path.join('test_files', 'test_fi'))

        with self.assertRaises(Exception):
            OpusRead(directory='Books', source='en',
                    target='fi', alignment_file=os.path.join(self.tempdir1,
                        'test_files', 'testlinks'),
                    source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
                    target_zip = os.path.join(self.tempdir1, 'test_fi.zip')
                ).printPairs()

    def test_try_to_open_wrongly_named_docs_from_specifed_target_zip(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp>+\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_en.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                arcname=os.path.join('test_files', 'test_en'))
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_fi.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                arcname=os.path.join('test_files', 'test_un'))

        with self.assertRaises(FileNotFoundError):
            OpusRead(directory='Books', source='en',
                    target='fi', alignment_file=os.path.join(self.tempdir1,
                        'test_files', 'testlinks'),
                    source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
                    target_zip = os.path.join(self.tempdir1, 'test_fi.zip')
                ).printPairs()

    def test_checks_first_whether_documents_are_in_path_gz(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en.gz" toDoc="test_files/test_fi.gz" >\n<link '
                'xtargets="s1;s1"/>\n </linkGrp>+\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s>\n </body>\n</text>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'rb') as f:
            with gzip.open(os.path.join(self.tempdir1, 'test_files',
                    'test_en.gz'), 'wb') as gf:
                shutil.copyfileobj(f, gf)
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'rb') as f:
            with gzip.open(os.path.join(self.tempdir1,
                    'test_files', 'test_fi.gz'), 'wb') as gf:
                shutil.copyfileobj(f, gf)

        var = pairPrinterToVariable(directory='Books', source='eo',
            target='pt', alignment_file=os.path.join(self.tempdir1,
                'test_files', 'testlinks'), download_dir=self.tempdir1,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# test_files/test_en.gz\n# test_files/test_fi.gz\n\n'
            '================================\n(src)="s1">test_en1 test_en2\n'
            '(trg)="s1">test_fi1 test_fi2'
            '\n================================\n')

    def test_filtering_by_src_cld2(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['en', '0.98'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s5.0">Mr. Sherlock Holmes'
            '\n(trg)="s5.0">Herra Sherlock Holmes'
            '\n================================\n')

    def test_filtering_by_trg_cld2(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, trg_cld2=['ia', '0'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s4">Chapter 1 Mr. Sherlock Holmes'
            '\n(trg)="s4">Herra Sherlock Holmes .'
            '\n================================\n')

    def test_filtering_by_src_langid(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_langid=['de', '0'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s167.0">" Excellent !'
            '\n(trg)="s167.0">" Erinomaista .'
            '\n================================\n')

    def test_filtering_by_trg_langid(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, trg_langid=['et', '0'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s4">Chapter 1 Mr. Sherlock Holmes'
            '\n(trg)="s4">Herra Sherlock Holmes .'
            '\n================================\n')

    def test_filtering_by_lang_labels(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['un', '0'],
            trg_cld2=['fi', '0.97'], src_langid=['en', '0.17'],
            trg_langid=['fi', '1'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s8.1">I believe'
            '\n(trg)="s8.1">Luulenpa että sinulla'
            '\n================================\n')

    def test_filtering_by_lang_labels_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['un', '0'],
            trg_cld2=['fi', '0.97'], src_langid=['en', '0.17'],
            trg_langid=['fi', '1'], fast=True,
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s8.1">I believe'
            '\n(trg)="s8.1">Luulenpa että sinulla'
            '\n================================\n')

    def test_filtering_by_lang_labels_nonalphabetical_lang_order(self):
        var = pairPrinterToVariable(directory='RF', source='sv', target='en',
            release='v1', maximum=1, trg_cld2=['un', '0'],
            src_cld2=['fi', '0.97'], trg_langid=['en', '0.17'],
            src_langid=['fi', '1'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s8.1">Luulenpa että sinulla'
            '\n(trg)="s8.1">I believe'
            '\n================================\n')

    def test_filtering_by_lang_labels_nonalphabetical_lang_order_fast(self):
        var = pairPrinterToVariable(directory='RF', source='sv', target='en',
            release='v1', maximum=1, trg_cld2=['un', '0'],
            src_cld2=['fi', '0.97'], trg_langid=['en', '0.17'],
            src_langid=['fi', '1'], fast=True,
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================'
            '\n(src)="s8.1">Luulenpa että sinulla'
            '\n(trg)="s8.1">I believe'
            '\n================================\n')

    def test_filtering_by_lang_labels_no_matches_found(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['fi', '2'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================\n')

    def test_filtering_by_lang_labels_no_matches_found_fast(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['fi', '2'], fast=True,
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz\n'
            '# sv/1996.xml.gz\n'
            '\n================================\n')

    def test_filtering_by_src_cld2_print_links(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['en', '0.98'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            write_mode='links', download_dir=self.tempdir1)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAli'
            'gn PUBLIC "-//CES//DTD XML cesAlign//EN" "">\n<cesAlign '
            'version="1.0">\n'
            '<linkGrp targType="s" fromDoc="en/1996.xml.gz" toDoc="sv'
            '/1996.xml.gz" >\n<link xtargets="s5.0;s5.0" id="SL5.0"/>'
            '\n </linkGrp>\n</cesAlign>\n')

    def test_filtering_by_lang_labels_print_links(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['un', '0'],
            trg_cld2=['fi', '0.97'], src_langid=['en', '0.17'],
            trg_langid=['fi', '1'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            write_mode='links', download_dir=self.tempdir1)
        self.assertEqual(var,
            '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAli'
            'gn PUBLIC "-//CES//DTD XML cesAlign//EN" "">\n<cesAlign '
            'version="1.0">\n'
            '<linkGrp targType="s" fromDoc="en/1996.xml.gz" toDoc="sv'
            '/1996.xml.gz" >\n<link xtargets="s8.1;s8.1" id="SL8.1"/>'
            '\n </linkGrp>\n</cesAlign>\n')

    def test_filtering_by_lang_labels_write_links(self):
        OpusRead(directory='RF', source='en', target='sv',
            release='v1', maximum=1, src_cld2=['un', '0'],
            trg_cld2=['fi', '0.97'], src_langid=['en', '0.17'],
            trg_langid=['fi', '1'],
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            write=[os.path.join(self.tempdir1, 'test_files', 'result')],
            write_mode='links', download_dir=self.tempdir1).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'result'),
                'r') as f:
            self.assertEqual(f.read(),
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAli'
                'gn PUBLIC "-//CES//DTD XML cesAlign//EN" "">\n<cesAlign '
                'version="1.0">\n'
                '<linkGrp targType="s" fromDoc="en/1996.xml.gz" toDoc="sv'
                '/1996.xml.gz" >\n<link xtargets="s8.1;s8.1" id="SL8.1"/>'
                '\n </linkGrp>\n</cesAlign>')

    def test_use_given_zip_files(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1, source_zip=os.path.join(self.tempdir1, 'en.zip'),
            target_zip=os.path.join(self.tempdir1, 'sv.zip'),
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1996.xml.gz'
            '\n# sv/1996.xml.gz'
            '\n\n================================'
            '\n(src)="s1">Source&<>"\' : manybooks.netAudiobook available here'
            '\n(trg)="s1">Source : Project Gutenberg'
            '\n================================\n')

    def test_use_given_zip_files_unalphabetical(self):
        var = pairPrinterToVariable(directory='RF', target='en', source='sv',
            maximum=1, target_zip=os.path.join(self.tempdir1, 'en.zip'),
            source_zip=os.path.join(self.tempdir1, 'sv.zip'),
            alignment_file=os.path.join(self.tempdir1, 'books_alignment.xml'),
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# en/1996.xml.gz'
            '\n# sv/1996.xml.gz'
            '\n\n================================'
            '\n(src)="s1">Source : Project Gutenberg'
            '\n(trg)="s1">Source&<>"\' : manybooks.netAudiobook available here'
            '\n================================\n')

    def test_source_zip_given_and_target_automatic(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            source_zip=os.path.join(self.tempdir1, 'en.zip'),
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.tempdir1, 'en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.root_directory, 'RF', 'latest', 'xml', 'sv.zip'))

    def test_source_zip_given_and_target_automatic_unalphabetical(self):
        opr = OpusRead(directory='RF', target='en', source='sv',
            source_zip=os.path.join(self.tempdir1, 'sv.zip'),
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.root_directory, 'RF', 'latest', 'xml', 'en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'sv.zip'))

    def test_target_zip_given_and_source_automatic(self):
        opr = OpusRead(directory='RF', source='en', target='sv',
            target_zip=os.path.join(self.tempdir1, 'sv.zip'),
            root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.root_directory, 'RF', 'latest', 'xml', 'en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'sv.zip'))

    def test_target_zip_given_and_source_local(self):
        opr = OpusRead(directory='RF', source='en', target='sv', release='v1',
            target_zip=os.path.join(self.tempdir1, 'sv.zip'),
            download_dir=self.tempdir1)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.tempdir1, 'RF_v1_xml_en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'sv.zip'))

    def test_target_zip_given_and_source_local_unalphabetical(self):
        opr = OpusRead(directory='RF', target='en', source='sv', release='v1',
            target_zip=os.path.join(self.tempdir1, 'en.zip'),
            download_dir=self.tempdir1)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.tempdir1, 'en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'RF_v1_xml_sv.zip'))

    def test_source_zip_given_and_target_local(self):
        opr = OpusRead(directory='RF', source='en', target='sv', release='v1',
            source_zip=os.path.join(self.tempdir1, 'en.zip'),
            download_dir=self.tempdir1)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.tempdir1, 'en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'RF_v1_xml_sv.zip'))

    def test_source_zip_local_and_target_automatic(self):
        opr = OpusRead(directory='RF', source='en', target='es', release='v1',
            download_dir=self.tempdir1, root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'en/1996.xml.gz', 'toDoc': 'es/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.tempdir1, 'RF_v1_xml_en.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.root_directory, 'RF', 'v1', 'xml', 'es.zip'))

    def test_source_zip_local_and_target_automatic_unalphabetical(self):
        opr = OpusRead(directory='RF', source='sv', target='es', release='v1',
            download_dir=self.tempdir1, root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'es/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.root_directory, 'RF', 'v1', 'xml', 'es.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'RF_v1_xml_sv.zip'))

    def test_target_zip_local_and_source_automatic(self):
        opr = OpusRead(directory='RF', source='es', target='sv', release='v1',
            download_dir=self.tempdir1, root_directory=self.root_directory)
        opr.par.initializeSentenceParsers(
            {'fromDoc': 'es/1996.xml.gz', 'toDoc': 'sv/1996.xml.gz'})
        self.assertEqual(opr.par.sourcezip.filename,
            os.path.join(self.root_directory, 'RF', 'v1', 'xml', 'es.zip'))
        self.assertEqual(opr.par.targetzip.filename,
            os.path.join(self.tempdir1, 'RF_v1_xml_sv.zip'))

    '''
    def test_empty_argument_list(self):
        temp_args = sys.argv.copy()
        arguments = '-d RF -s en -t sv -m 1 -f -rd'.split()
        arguments.append(self.root_directory)
        sys.argv = [temp_args[0]] + arguments
        var = pairPrinterToVariable([])
        self.assertEqual(var,
            '\n# en/1988.xml.gz\n# sv/1988.xml.gz\n\n================'
            '================\n(src)="s1.1">Statement of Government P'
            'olicy by the Prime Minister , Mr Ingvar Carlsson , at th'
            'e Opening of the Swedish Parliament on Tuesday , 4 Octob'
            'er , 1988 .\n(trg)="s1.1">REGERINGSFÖRKLARING .\n======='
            '=========================\n')
        sys.argv = temp_args.copy()
    '''

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_alignment_file_not_found(self, mocked_input):
        mocked_input.side_effect = ['y', 'n']
        opr = OpusRead(directory='RF', source='en', target='sv', maximum=1,
            alignment_file=os.path.join(self.tempdir1, 'unfound.xml.gz'),
            download_dir=self.tempdir1, root_directory=self.root_directory)
        opr.printPairs()
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en-sv.xml.gz'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en.zip'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_sv.zip'))
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            maximum=1,
            alignment_file=os.path.join(self.tempdir1, 'unfound.xml.gz'))
        self.assertTrue('No alignment file' in var)

    def test_alignment_file_not_found_no_prompt(self):
        opr = OpusRead(directory='RF', source='en', target='sv', maximum=1,
            alignment_file=os.path.join(self.tempdir1, 'unfound.xml.gz'),
            suppress_prompts=True, download_dir=self.tempdir1,
            root_directory=self.root_directory)
        opr.printPairs()
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir1,
            'RF_latest_xml_en-sv.xml.gz')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir1,
            'RF_latest_xml_en.zip')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir1,
            'RF_latest_xml_sv.zip')))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en-sv.xml.gz'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en.zip'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_sv.zip'))

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_zip_file_not_found(self, mocked_input):
        mocked_input.side_effect = ['y']
        opr = OpusRead(directory='RF', source='en', target='sv', maximum=1,
            download_dir=self.tempdir1, root_directory=self.root_directory)
        opr.par.source = ''

        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        opr.printPairs()
        sys.stdout = old_stdout

        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en-sv.xml.gz'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en.zip'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_sv.zip'))

        self.assertEqual(printout.getvalue()[-230:],
            '(src)="s1.1">S'
            'tatement of Government Policy by the Prime Minister , Mr'
            ' Ingvar Carlsson , at the Opening of the Swedish Parliame'
            'nt on Tuesday , 4 October , 1988 .\n(trg)="s1.1">REGERIN'
            'GSFÖRKLARING .\n================================\n')

    def test_testConfidence_with_empty_attrsList(self):
        self.assertFalse(self.opr.par.testConfidence('', [], ''))

    def test_id_file_printing(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            attribute='certainty', threshold='1',
            write_ids=os.path.join(self.tempdir1, 'test_files', 'test.id'),
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(self.tempdir1,
                'test_files', 'test.id')) as id_file:
            self.assertEqual(id_file.read(), 'en/1988.xml.gz\tsv/1988'
                '.xml.gz\ts3.2\ts3.2\t1.14214\n')

    def test_id_file_printing_unalphabetical(self):
        OpusRead(directory='RF', source='sv', target='en', maximum=1,
            src_range='1', tgt_range='2', attribute='certainty',
            threshold='0.1',
            write_ids=os.path.join(self.tempdir1, 'test_files', 'test.id'),
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(
                self.tempdir1, 'test_files', 'test.id')) as id_file:
            self.assertEqual(id_file.read(), 'sv/1988.xml.gz\ten/1988'
                '.xml.gz\ts4.4\ts4.4 s4.5\t0.188136\n')

    def test_id_file_printing_with_no_attribute(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            write_ids=os.path.join(self.tempdir1, 'test_files', 'test.id'),
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(
                self.tempdir1, 'test_files/test.id')) as id_file:
            self.assertEqual(id_file.read(), 'en/1988.xml.gz\tsv/1988'
                '.xml.gz\ts1.1\ts1.1\tNone\n')

    def test_id_file_printing_with_attribute_no_threshold(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            attribute='certainty',
            write_ids=os.path.join(self.tempdir1, 'test_files', 'test.id'),
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(
                self.tempdir1, 'test_files/test.id')) as id_file:
            self.assertEqual(id_file.read(), 'en/1988.xml.gz\tsv/1988'
                '.xml.gz\ts1.1\ts1.1\t-0.0636364\n')

    def test_id_file_printing_with_invalid_attribute(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            attribute='asfg',
            write_ids=os.path.join(self.tempdir1, 'test_files', 'test.id'),
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(
            self.tempdir1, 'test_files/test.id')) as id_file:
            self.assertEqual(id_file.read(), 'en/1988.xml.gz\tsv/1988'
                '.xml.gz\ts1.1\ts1.1\tNone\n')

    def test_id_file_printing_with_only_threshold(self):
        OpusRead(directory='RF', source='en', target='sv', maximum=1,
            threshold='0',
            write_ids=os.path.join(self.tempdir1, 'test_files', 'test.id'),
            root_directory=self.root_directory).printPairs()
        with open(os.path.join(
            self.tempdir1, 'test_files/test.id')) as id_file:
            self.assertEqual(id_file.read(), 'en/1988.xml.gz\tsv/1988'
                '.xml.gz\ts1.1\ts1.1\tNone\n')

    def test_pair_output_sending_with_single_output_file(self):
        self.opr.write_mode = 'moses'
        self.opr.write = [os.path.join(
            self.tempdir1, 'test_files', 'moses.txt')]
        self.opr.resultfile = open(self.opr.write[0], 'w')
        wpair = ('sentence 1\tsentence 2\n', '')
        self.opr.sendPairOutput(wpair)
        self.opr.resultfile.close()
        with open(os.path.join(
                self.tempdir1, 'test_files', 'moses.txt')) as mosesf:
            self.assertEqual(mosesf.read(), 'sentence 1\tsentence 2\n')

    def test_pair_output_sending_with_two_output_files(self):
        self.opr.write_mode = 'moses'
        self.opr.write = [
            os.path.join(self.tempdir1, 'test_files', 'moses.src'),
            os.path.join(self.tempdir1, 'test_files', 'moses.trg')]
        self.opr.mosessrc = open(self.opr.write[0], 'w')
        self.opr.mosestrg = open(self.opr.write[1], 'w')
        wpair = ('sentence 1\t', 'sentence 2\n')
        self.opr.sendPairOutput(wpair)
        self.opr.mosessrc.close()
        self.opr.mosestrg.close()
        with open(os.path.join(
                self.tempdir1, 'test_files', 'moses.src')) as mosessrc:
            self.assertEqual(mosessrc.read(), 'sentence 1\t')
        with open(os.path.join(
                self.tempdir1, 'test_files', 'moses.trg')) as mosestrg:
            self.assertEqual(mosestrg.read(), 'sentence 2\n')

    def test_writing_id_file_line(self):
        self.opr.id_file = open(os.path.join(
            self.tempdir1, 'test_files', 'id_file'), 'w')
        id_details = ('file_name1', 'file_name2',
            ['id1', 'id2'], ['id1'], 'value')
        self.opr.sendIdOutput(id_details)
        self.opr.id_file.close()
        with open(os.path.join(
                self.tempdir1, 'test_files', 'id_file')) as id_file:
            self.assertEqual(id_file.read(),
                'file_name1\tfile_name2\tid1 id2\tid1\tvalue\n')

    def test_writing_time_tags_xml(self):
        var = pairPrinterToVariable(directory='OpenSubtitles', source='eo',
            target='tl', maximum=1, preserve_inline_tags=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# eo/2009/1187043/6483790.xml.gz\n'
            '# tl/2009/1187043/6934998.xml.gz\n\n'
            '================================\n(src)="1"><time id="T1'
            'S" value="00:00:06,849" /> Ĉiuj nomoj , roluloj kaj evento'
            'j reprezentitaj en ĉi tiu filmo estas fikciaj .\n\n========'
            '========================\n')

    def test_writing_time_tags_xml_fast(self):
        var = pairPrinterToVariable(directory='OpenSubtitles', source='eo',
            target='tl', maximum=1, preserve_inline_tags=True, fast=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# eo/2009/1187043/6483790.xml.gz\n'
            '# tl/2009/1187043/6934998.xml.gz\n\n'
            '================================\n(src)="1"><time id="T1'
            'S" value="00:00:06,849" /> Ĉiuj nomoj , roluloj kaj evento'
            'j reprezentitaj en ĉi tiu filmo estas fikciaj .\n\n========'
            '========================\n')

    def test_writing_time_tags_raw(self):
        var = pairPrinterToVariable(directory='OpenSubtitles', source='eo',
            target='tl', maximum=1, preserve_inline_tags=True,
            preprocess='raw',
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# eo/2009/1187043/6483790.xml.gz\n'
            '# tl/2009/1187043/6934998.xml.gz\n\n'
            '================================\n(src)="1"><time id="T1'
            'S" value="00:00:06,849" />Ĉiuj nomoj, roluloj kaj evento'
            'j reprezentitaj en ĉi tiu filmo estas fikciaj.\n\n========'
            '========================\n')

    def test_writing_time_tags_raw_fast(self):
        var = pairPrinterToVariable(directory='OpenSubtitles', source='eo',
            target='tl', maximum=1, preserve_inline_tags=True,
            preprocess='raw',
            fast=True, root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# eo/2009/1187043/6483790.xml.gz\n'
            '# tl/2009/1187043/6934998.xml.gz\n\n'
            '================================\n(src)="1"><time id="T1'
            'S" value="00:00:06,849" />Ĉiuj nomoj, roluloj kaj evento'
            'j reprezentitaj en ĉi tiu filmo estas fikciaj.\n\n========'
            '========================\n')

    def test_addSentence_escapes_characters_when_write_mode_tmx(self):
        sentence = self.opr.par.sPar.addSentence('', ' &<>"\'', '1')
        self.assertEqual(sentence, '\n(src)="1"> &<>"\'')
        self.opr.par.sPar.wmode='tmx'
        sentence = self.opr.par.sPar.addSentence('', '&<>"\'', '')
        self.assertEqual(sentence, ' &amp;&lt;&gt;"\'')

    def test_escape_characters_when_write_mode_tmx(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, write_mode='tmx',
            download_dir=self.tempdir1,
            alignment_file=os.path.join(self.tempdir1,
                'books_alignment.xml'))
        self.assertEqual(var, '<?xml version="1.0" encoding="utf-8"?>'
            '\n<tmx version="1.4.">\n<header srclang="en"'
            '\n\tadminlang="en"\n\tsegtype="sentence"'
            '\n\tdatatype="PlainText" />\n\t<body>\n\t\t<tu>'
            '\n\t\t\t<tuv xml:lang="en"><seg>Source&amp;&lt;&gt;"\' : '
            'manybooks.netAudiobook available here</seg></tuv>'
            '\n\t\t\t<tuv xml:lang="sv"><seg>Source : Project Gutenberg</seg>'
            '</tuv>\n\t\t</tu>\n\t</body>\n</tmx>\n')

    def test_open_predownloaded_alignment_file(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            release='v1', maximum=1, download_dir=self.tempdir1)
        self.assertEqual(var,
            '\n# en/1996.xml.gz'
            '\n# sv/1996.xml.gz'
            '\n\n================================'
            '\n(src)="s1">Source&<>"\' : manybooks.netAudiobook available here'
            '\n(trg)="s1">Source : Project Gutenberg'
            '\n================================\n')

    def test_no_plain_xml_alingment_file_found(self):
        var = pairPrinterToVariable(directory='RF', source='en', target='sv',
            alignment_file='unfound.xml')
        self.assertEqual(var, 'No alignment file "unfound.xml" found\n')

    def test_download_zip_files_no_prompt(self):
        var = pairPrinterToVariable(directory='RF', source='fr', target='sv',
            maximum=1, download_dir=self.tempdir1, suppress_prompts=True,
            root_directory=self.root_directory)
        self.assertEqual(var[-230:],
            '(src)="s1.1">Declaration de Politique Générale du '
            'Gouvernement présentée mardi 4 octobre 1988 devant le '
            'Riksdag par Monsieur Ingvar Carlsson , Premier Ministre .\n'
            '(trg)="s1.1">REGERINGSFÖRKLARING .\n'
            '================================\n')
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_fr.zip'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_sv.zip'))
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_fr-sv.xml.gz'))

    def test_alignment_file_could_not_be_parsed(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp></linkGrp>\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_en.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                arcname=os.path.join('test_files', 'test_en'))
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_fi.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                arcname=os.path.join('test_files', 'test_fi'))

        with self.assertRaises(AlignmentParserError):
            OpusRead(directory='Books', source='en',
                    target='fi', alignment_file=os.path.join(self.tempdir1,
                        'test_files', 'testlinks'),
                    source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
                    target_zip = os.path.join(self.tempdir1, 'test_fi.zip')
                ).printPairs()
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp\n</cesAlign>')
        with self.assertRaises(AlignmentParserError):
            OpusRead(directory='Books', source='en',
                    target='fi', alignment_file=os.path.join(self.tempdir1,
                        'test_files', 'testlinks'),
                    source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
                    target_zip = os.path.join(self.tempdir1, 'test_fi.zip')
                ).printPairs()

    def test_sentence_file_could_not_be_parsed(self):
        with open(os.path.join(self.tempdir1, 'test_files', 'testlinks'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE cesAlign '
                'PUBLIC "-//CES//DTD XML cesAlign//EN" "">'
                '\n<cesAlign version="1.0">\n<linkGrp fromDoc="test_files/'
                'test_en" toDoc="test_files/test_fi" >\n<link xtargets='
                '"s1;s1"/>\n </linkGrp>\n</cesAlign>')
        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s><s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_en.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                arcname=os.path.join('test_files', 'test_en'))
        with open(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n <body>\n'
                '<s id="s1">\n <w>test_fi1</w>\n <w>test_fi2'
                '</w>\n</s>\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_fi.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_fi'),
                arcname=os.path.join('test_files', 'test_fi'))

        with self.assertRaises(SentenceParserError):
            OpusRead(directory='Books', source='en',
                    target='fi', alignment_file=os.path.join(self.tempdir1,
                        'test_files', 'testlinks'),
                    source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
                    target_zip = os.path.join(self.tempdir1, 'test_fi.zip')
                ).printPairs()

        with open(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                'w') as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8"?>\n<text>\n'
                '<body>\n<s id="s1">\n <w>test_en1</w>\n <w>test_en2'
                '</w>\n</s\n </body>\n</text>')
        with zipfile.ZipFile(os.path.join(self.tempdir1, 'test_en.zip'),
                'w') as zf:
            zf.write(os.path.join(self.tempdir1, 'test_files', 'test_en'),
                arcname=os.path.join('test_files', 'test_en'))

        with self.assertRaises(SentenceParserError):
            OpusRead(directory='Books', source='en',
                    target='fi', alignment_file=os.path.join(self.tempdir1,
                        'test_files', 'testlinks'),
                    source_zip = os.path.join(self.tempdir1, 'test_en.zip'),
                    target_zip = os.path.join(self.tempdir1, 'test_fi.zip')
                ).printPairs()


class TestOpusCat(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.maxDiff = None

        if ('OPUSCAT_TEST_TEMPDIR' in os.environ.keys() and
            os.path.exists(os.environ['OPUSCAT_TEST_TEMPDIR'])):
                self.tempdir1 = os.environ['OPUSCAT_TEST_TEMPDIR']
        else:
            self.tempdir1 = tempfile.mkdtemp()
            os.mkdir(os.path.join(self.tempdir1, 'test_files'))

        if ('OPUSCAT_TEST_ROOTDIR' in os.environ.keys() and
            os.path.exists(os.environ['OPUSCAT_TEST_ROOTDIR'])):
            self.root_directory = os.environ['OPUSCAT_TEST_ROOTDIR']
        else:
            self.root_directory = tempfile.mkdtemp()

            os.makedirs(os.path.join(self.root_directory, 'RF', 'latest',
                'xml'))

            add_to_root_dir(corpus='RF', source='en', target='sv',
                preprocess='xml', root_dir=self.root_directory)

    @classmethod
    def tearDownClass(self):
        if ('OPUSCAT_TEST_SAVE' in os.environ.keys() and
                os.environ['OPUSCAT_TEST_SAVE'] == 'true'):
            print('\nTEMPDIR:', self.tempdir1)
            print('ROOTDIR:', self.root_directory)
        else:
            shutil.rmtree(self.tempdir1)
            shutil.rmtree(self.root_directory)

    def printSentencesToVariable(self, **kwargs):
        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        oprinter = OpusCat(**kwargs)
        oprinter.printSentences()
        sys.stdout = old_stdout
        return printout.getvalue()

    def test_printing_sentences(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            plain=True, root_directory=self.root_directory)
        self.assertEqual(var[-183:],
            """("s72.1")>It is the Government 's resposibility and ai"""
            """m to put to use all good initiatives , to work for bro"""
            """ad solutions and to pursue a policy in the interests o"""
            """f the whole nation .\n""")

    def test_printing_sentences_with_limit(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            maximum=1, plain=True, root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# RF/xml/en/1996.xml\n\n("s1.1")>MINISTRY FOR FOREIGN'
            ' AFFAIRS Press Section Check against delivery\n')

    def test_printing_sentences_without_ids(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            maximum=1, no_ids=True, plain=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# RF/xml/en/1996.xml\n\nMINISTRY FOR FOREIGN'
            ' AFFAIRS Press Section Check against delivery\n')

    def test_print_annotations(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            maximum=1, no_ids=True, plain=True, print_annotations=True,
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# RF/xml/en/1996.xml\n\nMINISTRY|NNP|ministry FOR|NNP'
            '|for FOREIGN|NNP|FOREIGN AFFAIRS|NNP Press|NNP|Press Sec'
            'tion|NNP|Section Check|NNP|Check against|IN|against deli'
            'very|NN|delivery\n')

    def test_print_annotations_all_attributes(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            maximum=1, no_ids=True, plain=True, print_annotations=True,
            set_attribute=['all_attrs'], root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# RF/xml/en/1996.xml\n\nMINISTRY|null|0|NN|w1.1.1|mini'
            'stry|NNP|NN FOR|prep|1|IN|w1.1.2|for|NNP|IN FOREIGN|nn|7'
            '|NNP|w1.1.3|FOREIGN|NNP|NP AFFAIRS|nn|7|NNP|w1.1.4|NNP|N'
            'P Press|nn|7|NNP|w1.1.5|Press|NNP|NP Section|nn|7|NNP|w1'
            '.1.6|Section|NNP|NP Check|pobj|2|NNP|w1.1.7|Check|NNP|NP'
            ' against|prep|1|IN|w1.1.8|against|IN|IN delivery|pobj|8|N'
            'N|w1.1.9|delivery|NN|NN\n')

    def test_print_xml(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            maximum=1, root_directory=self.root_directory)
        self.assertEqual(var[-38:],
            '<w id="w2.10">1996</w>\n</p><p id="3">\n')

    def test_printing_specific_file(self):
        var = self.printSentencesToVariable(directory='RF', language='en',
            maximum=1, no_ids=True, plain=True, file_name='RF/xml/en/1988.xml',
            root_directory=self.root_directory)
        self.assertEqual(var,
            '\n# RF/xml/en/1988.xml\n\nStatement of Government Policy'
            ' by the Prime Minister , Mr Ingvar Carlsson , at the Open'
            'ing of the Swedish Parliament on Tuesday , 4 October , 1'
            '988 .\n')

    '''
    def test_empty_argument_list(self):
        temp_args = sys.argv.copy()
        sys.argv = [temp_args[0]] + '-d RF -l en -m 1 -p -rd {rootdir}'.format(
            rootdir=self.root_directory).split()
        var = self.printSentencesToVariable([])
        self.assertEqual(var,
            '\n# RF/xml/en/1996.xml\n\n("s1.1")>MINISTRY FOR FOREIGN '
            'AFFAIRS Press Section Check against delivery\n')
        sys.argv = temp_args.copy()
    '''

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_file_not_found(self, mocked_input):
        mocked_input.side_effect = ['y']
        var = self.printSentencesToVariable(directory='RFOSIAJ', language='en',
            maximum=1, plain=True, root_directory=self.root_directory)

        self.assertEqual(var[-28:],
            '\nNecessary files not found.\n')

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_download_necessary_files(self, mocked_input):
        mocked_input.side_effect = ['y', 'n', 'n']

        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        OpusCat.openFiles(
            OpusCat(directory='RF', language='en', download_dir=self.tempdir1),
            os.path.join(self.tempdir1, 'RF_latest_xml_en.zip'), '')
        os.remove(os.path.join(self.tempdir1, 'RF_latest_xml_en.zip'))
        OpusCat.openFiles(
            OpusCat(directory='RF', language='en', download_dir=self.tempdir1),
            os.path.join(self.tempdir1, 'RF_latest_xml_en.zip'), '')
        sys.stdout = old_stdout
        print(printout.getvalue())
        self.assertTrue('No file found' in printout.getvalue())

class TestOpusGet(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        self.maxDiff = None

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    '''
    def test_empty_argument_list(self):
        temp_args = sys.argv.copy()
        sys.argv = [temp_args[0]] + '-s eo'.split()
        opg = OpusGet([])
        params = ['source=eo', 'version=latest', 'preprocessing=xml', '']
        for param in opg.url.split('?')[1].split('&'):
            self.assertTrue(param in params)
        sys.argv = temp_args.copy()
    '''

    def test_format_size(self):
        opg = OpusGet(source='eo')
        self.assertEqual(opg.format_size(1), '1 KB')
        self.assertEqual(opg.format_size(291), '291 KB')
        self.assertEqual(opg.format_size(1000), '1 MB')
        self.assertEqual(opg.format_size(2514), '3 MB')
        self.assertEqual(opg.format_size(1000000), '1 GB')
        self.assertEqual(opg.format_size(3385993), '3 GB')
        self.assertEqual(opg.format_size(1000000000), '1 TB')
        self.assertEqual(opg.format_size(2304006273), '2 TB')

    def test_remove_data_with_no_alignment(self):
        opg = OpusGet(source='en', target='sv', list_resources=True)
        data = {'corpora':
                [{'alignment_pairs': 219,
                    'corpus': 'RF',
                    'documents': 2,
                    'id': 321123,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 4,
                    'source': 'en',
                    'source_tokens': 4393,
                    'target': 'sv',
                    'target_tokens': 2905,
                    'url': ('https://object.pouta.csc.fi/OPUS-RF/v1/xml/'
                        'en-sv.xml.gz'),
                    'version': 'v1'},
                {'alignment_pairs': 181,
                    'corpus': 'RF',
                    'documents': 2,
                    'id': 321124,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 60,
                    'source': 'en',
                    'source_tokens': 4393,
                    'target': '',
                    'target_tokens': None,
                    'url': 'https://object.pouta.csc.fi/OPUS-RF/v1/xml/en.zip',
                    'version': 'v1'},
                {'alignment_pairs': 298,
                    'corpus': 'RF',
                    'documents': 4,
                    'id': 321130,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 64,
                    'source': 'sv',
                    'source_tokens': 3456,
                    'target': '',
                    'target_tokens': None,
                    'url': 'https://object.pouta.csc.fi/OPUS-RF/v1/xml/sv.zip',
                    'version': 'v1'},
                {'alignment_pairs': 181,
                    'corpus': 'Test',
                    'documents': 2,
                    'id': 321124,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 60,
                    'source': 'en',
                    'source_tokens': 4393,
                    'target': '',
                    'target_tokens': None,
                    'url': 'https://object.pouta.csc.fi/OPUS-T/v1/xml/en.zip',
                    'version': 'v1'},
                {'alignment_pairs': 298,
                    'corpus': 'Test',
                    'documents': 4,
                    'id': 321130,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 64,
                    'source': 'sv',
                    'source_tokens': 3456,
                    'target': '',
                    'target_tokens': None,
                    'url': 'https://object.pouta.csc.fi/OPUS-T/v1/xml/sv.zip',
                    'version': 'v1'}]}

        new_data = opg.remove_data_with_no_alignment(data)
        self.assertEqual(len(new_data), 3)
        data['corpora'].pop(0)
        new_data = opg.remove_data_with_no_alignment(data)
        self.assertEqual(len(new_data), 0)

    def test_add_data_with_alignment(self):
        opg = OpusGet(directory= 'RF', source='en', target='sv',
            list_resources=True)
        data = [{'alignment_pairs': 219,
                    'corpus': 'RF',
                    'documents': 2,
                    'id': 321123,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 4,
                    'source': 'en',
                    'source_tokens': 4393,
                    'target': 'sv',
                    'target_tokens': 2905,
                    'url': ('https://object.pouta.csc.fi/OPUS-RF/v1/xml/'
                        'en-sv.xml.gz'),
                    'version': 'v1'},
                {'alignment_pairs': 181,
                    'corpus': 'RF',
                    'documents': 2,
                    'id': 321124,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 60,
                    'source': 'en',
                    'source_tokens': 4393,
                    'target': '',
                    'target_tokens': None,
                    'url': 'https://object.pouta.csc.fi/OPUS-RF/v1/xml/en.zip',
                    'version': 'v1'},
                {'alignment_pairs': 298,
                    'corpus': 'RF',
                    'documents': 4,
                    'id': 321130,
                    'latest': 'True',
                    'preprocessing': 'xml',
                    'size': 64,
                    'source': 'sv',
                    'source_tokens': 3456,
                    'target': '',
                    'target_tokens': None,
                    'url': 'https://object.pouta.csc.fi/OPUS-RF/v1/xml/sv.zip',
                    'version': 'v1'}]
        new_data = opg.add_data_with_aligment(data, [])
        self.assertEqual(len(new_data), 3)
        data.pop(0)
        new_data = opg.add_data_with_aligment(data, [])
        self.assertEqual(len(new_data),0 )

    def test_remove_data_with_no_alignment_from_OPUS(self):
        opg = OpusGet(source='en', target='sv', list_resources=True)
        data = opg.get_response(opg.url)
        new_data = opg.remove_data_with_no_alignment(data)
        self.assertLess(len(new_data), len(data['corpora']))

    def test_get_files_invalid_url(self):
        opg = OpusGet(directory='RF', source='en', target='sv',
            list_resources=True)
        opg.url = 'http://slkdfjlks'
        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        opg.get_files()
        sys.stdout = old_stdout

        self.assertEqual(printout.getvalue(), 'Unable to retrieve the data.\n')

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_download_invalid_url(self, mocked_input):
        mocked_input.side_effect = ['y']
        opg = OpusGet(directory='RF', source='en', target='sv',
            list_resources=True)
        corpora, file_n, total_size = opg.get_corpora_data()
        corpora[0]['url'] = 'http://alskdjfl'
        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        opg.download(corpora, file_n, total_size)
        sys.stdout = old_stdout

        self.assertEqual(printout.getvalue(), 'Unable to retrieve the data.\n')

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_dont_list_files_that_are_already_in_path(self, mocked_input):
        mocked_input.side_effect = ['y']
        OpusGet(directory='RF', source='en', target='sv', preprocess='xml',
            download_dir=self.tempdir).get_files()
        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        OpusGet(directory='RF', source='en', target='sv', preprocess='xml',
            download_dir=self.tempdir, list_resources=True).get_files()
        sys.stdout = old_stdout
        os.remove(os.path.join(self.tempdir, 'RF_latest_xml_en-sv.xml.gz'))
        os.remove(os.path.join(self.tempdir, 'RF_latest_xml_en.zip'))
        os.remove(os.path.join(self.tempdir, 'RF_latest_xml_sv.zip'))

        self.assertEqual(printout.getvalue(),
            '        {tempdir}/RF_latest_xml_en-sv.xml.gz already exi'
            'sts\n        {tempdir}/RF_latest_xml_en.zip already exis'
            'ts\n        {tempdir}/RF_latest_xml_sv.zip already exist'
            's\n\n   0 KB Total size\n'.format(tempdir=self.tempdir))

    @mock.patch('opustools_pkg.opus_get.input', create=True)
    def test_dont_download_files_that_are_already_in_path(self, mocked_input):
        mocked_input.side_effect = ['y', 'y']
        OpusGet(directory='RF', source='en', target='sv', preprocess='xml',
            download_dir=self.tempdir).get_files()
        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        OpusGet(directory='RF', source='en', target='sv', preprocess='xml',
            download_dir=self.tempdir).get_files()
        sys.stdout = old_stdout
        os.remove(os.path.join(self.tempdir, 'RF_latest_xml_en-sv.xml.gz'))
        os.remove(os.path.join(self.tempdir, 'RF_latest_xml_en.zip'))
        os.remove(os.path.join(self.tempdir, 'RF_latest_xml_sv.zip'))

        self.assertEqual(printout.getvalue(), '')

    def test_download_everything_from_a_corpus(self):
        old_stdout = sys.stdout
        printout = io.StringIO()
        sys.stdout = printout
        files = OpusGet(directory='RF', release='v1', preprocess='xml',
            list_resources=True).get_files()
        sys.stdout = old_stdout
        self.assertEqual(len(printout.getvalue().split('\n')), 18)

if __name__ == '__main__':
    unittest.main()

