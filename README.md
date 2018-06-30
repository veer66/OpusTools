# OpusTools

Tools for accessing and processing OPUS data.

* opus-tools: general python library of OPUS-related tools as class library
* opus-read: read parallel data sets and convert to different output formats
* opus-cat: extract given OPUS document from release data
* opus-api: web-service that provides information about OPUS data and download URLs
* opus-get: download parallel corpora
* opus-find: search for resources

## opus-get

* download data files from OPUS (ObjectStorage)
* allow to download several corpus files (from different corpora) - Do we need to restrict it somehow?
* store statistics about downloads
* command-line options
  * -s <source-lang-ID> (required)
  * -t <target-lang-ID> (optional?)
  * -c <corpus> (optional)
  * -f <data format> (optional, default = tokenized xml + sentalign-xml)
 


## opus-read related task- and wish-list:

* a python library of data access functions
  * status: to do
  * reader and writer classes for data in different formats
  * installable package (pip package?)
  
* OPUS reader
  * status: started
  * read OPUS corpora in their native XML formats (sentence-aligned)
  * obligatory parameters: corpus name, source and target language
  * otional parameters: release version, type (raw, xml, parsed), opus root-dir
  
* filter on alignment type and link likelihood
  * status: started
  * extract certain alignment types (for example only one-to-one sentence alignments or even ranges like 1-2 sentences in source aligned to one sentence in target etc)
  * filter on alignment certainty (if the attribute is given in the sentence alignment file)
  * filter on other attributes (e.g. overlap in OpenSubtitles)
  * possible flag to output only the sentence alignments information after filtering (in XML format)
  
* Moses writer
  * status: started
  * output in Moses format (2 files with aligned sentence on the same line)
  * skip empty alignments (source sentences or target sentences that are not aligned) - using flags like in filtering mode (see above)

* TMX writer
  * status: to do
  * output in TMX format
  * possibly: even for more than 2 aligned languages? (that would require transitive expansion of sentence alignments - difficult task - should we define that as a separate task/feature?)

* output with (token) annotation (factors)
  * extract token factors from tokenized corpora besides of the plain text (part-of-speech tags etc) if they exist
  * this only works with tokenized corpora
  * format like in factored Moses? Important: same number of factors for each token in that format! (possibility to change the factor delimiter symbol - default = '|'), example:
~~~
word1|POS1|lemma1 word2|POS2|lemma2
~~~

* OPUS root can be changed
  * status: to do
  * OPUS root directory is hard-coded now - make it more flexible
  * option to change it
  * possibly work automatically on Abel (Norwegian cluster) without changing with the optional flag
  
* read parallel corpora that are not packed into zip-files
  * status: to do
  * can read given sentence alignment file
  * checks whether aligned xml documents exist in the path before trying to find the zipped OPUS release files

* simple cat function (opus-cat)
  * status: to do
  * read a file from a given corpus in a given language (extract from corresponding zip file)
