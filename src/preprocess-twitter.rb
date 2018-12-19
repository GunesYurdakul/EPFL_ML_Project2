# Ruby 2.0
# Reads stdin: ruby -n preprocess-twitter.rb
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington

STDOUT.reopen(File.open('deneme_test.txt', 'w+'))

def tokenize input

	# Different regex parts for smiley faces
	eyes = "[8:=;]"
	nose = "['`\-]?"

	input = input
		.gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<URL>")
		.gsub("/"," / ") # Force splitting words appended with slashes (once we tokenized the URLs, of course)
		.gsub(/@\w+/, "<USER>")
		.gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<SMILE>")
		.gsub(/#{eyes}#{nose}p+/i, "<LOLFACE>")
		.gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<SADFACE>")
		.gsub(/#{eyes}#{nose}[\/|l*]/, "<NEUTRALFACE>")
		.gsub(/<3/,"<HEART>")
		.gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			"#{$~[1]}"
		}
		.gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
			# TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
			$~[1] + $~[2]
		}

	return input
end

File.open("test_data.txt", "r") do |f|
  f.each_line do |line|
    puts tokenize(line)
  end
end
