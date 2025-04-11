source "https://rubygems.org"

# Use GitHub Pages - this will constrain all dependencies to versions that work on GitHub Pages
gem "github-pages", "~> 232", group: :jekyll_plugins

# Add additional plugins
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-seo-tag", "~> 2.7"
end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
gem "tzinfo-data", platforms: [:mingw, :mswin, :x64_mingw, :jruby]

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock webrick for Ruby 3.0 compatibility
gem "webrick", "~> 1.7" 