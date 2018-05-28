import xml.parsers.expat

class ExhaustiveSentenceParser:

	def __init__(self, document, preprocessing, direction, wmode, language):
		self.document = document
		self.pre = preprocessing
		self.direction = direction
		self.wmode = wmode
		self.language = language

		self.start = ""
		self.sid = ""
		self.chara = ""
		self.end = ""

		self.sfound = False
		self.efound = False

		self.parser = xml.parsers.expat.ParserCreate()
		self.parser.StartElementHandler = self.start_element
		self.parser.CharacterDataHandler = self.char_data
		self.parser.EndElementHandler = self.end_element

		self.sentences = {}
		self.done = False

	def start_element(self, name, attrs):
		self.start = name
		if name == "s" and "id" in attrs.keys():
			self.sid = attrs["id"]
			self.sfound = True

	def char_data(self, data):
		self.chara = repr(data)

	def end_element(self, name):
		self.end = name
		if name == "s":
			self.efound = True

	def parseLine(self, line):
		self.parser.Parse(line.strip())

	def storeSentences(self):
		self.document.readline()
		while not self.done:
			sentence = ""
			while True:
				line = self.document.readline()
				if not line:
					self.done = True
					break
				self.parseLine(line)
				if self.pre == "xml":
					if self.sfound and self.start == "w" and self.end == "w":
						sentence = sentence + " " + self.chara[1:-1]
				elif self.pre == "raw":
					if self.sfound and self.start == "s":
						sentence = self.chara[:-1]
				elif self.pre == "rawos":
					if self.sfound and self.start == "time" or self.start == "s":
						sentence = self.chara[:-1]
				if self.efound:
					self.sfound = False
					self.efound = False
					break
			if self.sid != "":
				self.sentences[self.sid] = sentence[1:]
				self.sid = ""
		self.document.close()
		
	def getSentence(self, sid):
		if sid in self.sentences.keys():
			return self.sentences[sid]
		else:
			return ""

	def addTuBeginning(self):
		sentences = " "
		if self.direction == "src":
			sentences = sentences + "\t\t<tu>\n"
		sentences = sentences + '\t\t\t<tuv xml:lang="'+self.language+'"><seg>'
		return sentences

	def addTuEnding(self, sentences):
		sentences = sentences + "</seg></tuv>"
		if self.direction == "trg":
			sentences = sentences + "\n\t\t</tu>"
		return sentences

	def readSentence(self, ids):
		if len(ids) == 0 or ids[0] == "":
			return ""
		sentence = ""
		if self.wmode == "tmx":
			sentence = self.addTuBeginning()
		for sid in ids:
			if self.wmode == "normal":
				sentence = sentence + '\n('+self.direction+')="'+sid+'">'+self.getSentence(sid)
			elif self.wmode == "moses" or self.wmode == "tmx":
				sentence = sentence + " " + self.getSentence(sid)
				sentence = sentence.replace("<seg> ", "<seg>")
		if self.wmode == "tmx":
			sentence = self.addTuEnding(sentence)
				
		return sentence[1:]