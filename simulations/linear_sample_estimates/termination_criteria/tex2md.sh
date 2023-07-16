pandoc -s termination_criteria.tex -o termination_criteria.md
perl -pi -e 's/_/\\_/g' termination_criteria.md
perl -pi -e 's/\|/\\|/g' termination_criteria.md

