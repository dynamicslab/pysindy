#!/usr/bin/ruby

# For an OO language, this is distinctly procedural. Should probably fix that.
require 'json'

details = Hash.new({})

capture_params = [
  { :name => "title", :message => "Enter project name." },
  { :name => "url", :message => "Enter the URL of the project repository." },
  { :name => "description", :message => "Enter the (short) project description." },
  { :name => "license", :message => "Enter the license this software shared under. (hit enter to skip)\nFor example MIT, BSD, GPL v3.0, Apache 2.0" },
  { :name => "doi", :message => "Enter the DOI of the archived version of this code. (hit enter to skip)\nFor example http://dx.doi.org/10.6084/m9.figshare.828487" },
  { :name => "keywords", :message => "Enter keywords that should be associated with this project (hit enter to skip)\nComma-separated, for example: turkey, chicken, pot pie" },
  { :name => "version", :message => "Enter the version of your software (hit enter to skip)\nSEMVER preferred: http://semver.org e.g. v1.0.0" }
]

puts "I'm going to try and help you prepare some things for your JOSS submission"
puts "If all goes well then we'll have a nice codemeta.json file soon..."
puts ""
puts "************************************"
puts "*    First, some basic details     *"
puts "************************************"
puts ""

# Loop through the desired captures and print out for clarity
capture_params.each do |param|
  puts param[:message]
  print "> "
  input = gets

  details[param[:name]] = input.chomp

  puts ""
  puts "OK, your project has #{param[:name]}: #{input}"
  puts ""
end

puts ""
puts "************************************"
puts "*        Experimental stuff        *"
puts "************************************"
puts ""

puts "Would you like me to try and build a list of authors for you?"
puts "(You need to be running this script in a git repository for this to work)"
print "> (Y/N)"
answer = gets.chomp

case answer.downcase
when "y", "yes"

  # Use git shortlog to extract a list of author names and commit counts.
  # Note we don't extract emails here as there's often different emails for
  # each user. Instead we capture emails at the end.

  git_log = `git shortlog --summary --numbered --no-merges`

  # ["252\tMichael Jackson", "151\tMC Hammer"]
  authors_and_counts = git_log.split("\n").map(&:strip)

  authors_and_counts.each do |author_count|
    count, author = author_count.split("\t").map(&:strip)

    puts "Looks like #{author} made #{count} commits"
    puts "Add them to the output?"
    print "> (Y/N)"
    answer = gets.chomp

    # If a user chooses to add this author to the output then we ask for some
    # additional information including their email, ORCID and affiliation.
    case answer.downcase
    when "y", "yes"
      puts "What is #{author}'s email address? (hit enter to skip)"
      print "> "
      email = gets.chomp

      puts "What is #{author}'s ORCID? (hit enter to skip)"
      puts "For example: http://orcid.org/0000-0000-0000-0000"
      print "> "
      orcid = gets.chomp

      puts "What is #{author}'s affiliation? (hit enter to skip)"
      print "> "
      affiliation = gets.chomp


      details['authors'].merge!(author => { 'commits' => count,
                                            'email' => email,
                                            'orcid' => orcid,
                                            'affiliation' => affiliation })

    when "n", "no"
      puts "OK boss..."
      puts ""
    end
  end
when "n", "no"
  puts "OK boss..."
  puts ""
end

puts "Reticulating splines"

5.times do
  print "."
  sleep 0.5
end

puts ""
puts "Generating some JSON goodness..."

# TODO: work out how to use some kind of JSON template here.
# Build the output list of authors from the inputs we've collected.
output_authors = []

details['authors'].each do |author_name, values|
  entry = {
    "@id" => values['orcid'],
    "@type" => "Person",
    "email" => values['email'],
    "name" => author_name,
    "affiliation" => values['affiliation']
  }
  output_authors << entry
end

# TODO: this is currently a static template (written out here). It would be good
# to do something smarter here.
output = {
  "@context" => "https://raw.githubusercontent.com/codemeta/codemeta/master/codemeta.jsonld",
  "@type" => "Code",
  "author" => output_authors,
  "identifier" => details['doi'],
  "codeRepository" => details['url'],
  "datePublished" => Time.now.strftime("%Y-%m-%d"),
  "dateModified" => Time.now.strftime("%Y-%m-%d"),
  "dateCreated" => Time.now.strftime("%Y-%m-%d"),
  "description" => details['description'],
  "keywords" => details['keywords'],
  "license" => details['license'],
  "title" => details['title'],
  "version" => details['version']
}

File.open('codemeta.json', 'w') {|f| f.write(JSON.pretty_generate(output)) }