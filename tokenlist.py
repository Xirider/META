stopper = [ 
"<tr>",
"<td>",
"[newline]",
"[h1]",
"[h2]",
"[h3]",
"[h4]",
"[h5]",
"[h6]",
"[imagestart]",
"[imageend]",
"[unorderedlist]",
"[orderedlist]",
"[codestart]",
"[codeend]",
"[linkstart]",
"[linkend]" ,
"[CLS]",
"[SEP]",
"[Q]",
"[segment=00]", 
"[segment=01]" , 
"[segment=02]" ,
"[segment=03]" , 
"[segment=04]" , 
"[segment=05]" , 
"[segment=06]" , 
"[segment=07]" , 
"[segment=08]" ,
"[segment=09]" , 
"[segment=10]" , 
"[segment=11]" , 
"[segment=12]" , 
"[segment=13]" , 
"[segment=14]" ,
"[segment=15]" , 
"[segment=16]" , 
"[segment=17]" , 
"[segment=18]" , 
"[segment=19]" , 
"[segment=20]" , 
"[segment=21]" , 
"[segment=22]" , 
"[segment=23]" , 
"[segment=24]" , 
"[segment=25]" , 
"[segment=26]" , 
"[segment=27]" , 
"[segment=28]" , 
"[segment=29]" , 
"[segment=30]" , 
"[segment=xx]",
]

segment_tokens = ["[segment=00]" , "[segment=01]" , "[segment=02]" ,
        "[segment=03]" , "[segment=04]" , "[segment=05]" , "[segment=06]" , "[segment=07]" , "[segment=08]" ,
        "[segment=09]" , "[segment=10]" , "[segment=11]" , "[segment=12]" , "[segment=13]" , "[segment=14]" ,
        "[segment=15]" , "[segment=16]" , "[segment=17]" , "[segment=18]" , "[segment=19]" , "[segment=20]" , 
        "[segment=21]" , "[segment=22]" , "[segment=23]" , "[segment=24]" , "[segment=25]" , "[segment=26]" , 
        "[segment=27]" , "[segment=28]" , "[segment=29]" , "[segment=30]" , "[segment=xx]"]


headline_tokens = ["[h1]",
"[h2]",
"[h3]",
"[h4]",
"[h5]",
"[h6]",]

# stopper = ["[Newline]" , "[UNK]" , "[SEP]" , "[Q]" , "[CLS]" , "[WebLinkStart]" , "[LocalLinkStart]" , "[RelativeLinkStart]" ,
#     "[WebLinkEnd]" , "[LocalLinkEnd]" , "[RelativeLinkEnd]" , "[VideoStart]" , "[VideoEnd]" , "[TitleStart]" , 
#     "[NavStart]" , "[AsideStart]" , "[FooterStart]" , "[IframeStart]" , "[IframeEnd]" , "[NavEnd]" , "[AsideEnd]" , 
#     "[FooterEnd]" , "[CodeStart]" , "[H1Start]" , "[H2Start]" , "[H3Start]" , "[H4Start]" , "[H5Start]" , "[H6Start]" ,
#     "[CodeEnd]" , "[UnorderedList=1]" , "[UnorderedList=2]" , "[UnorderedList=3]" , "[UnorderedList=4]" , "[OrderedList]"
#     , "[UnorderedListEnd=1]" , "[UnorderedListEnd=2]" , "[UnorderedListEnd=3]" , "[UnorderedListEnd=4]" , 
#     "[OrderedListEnd]" , "[TableStart]" , "[RowStart]" , "[CellStart]" , "[TableEnd]" , "[RowEnd]" , "[CellEnd]" ,
#     "[LineBreak]" , "[Paragraph]" , "[StartImage]" , "[EndImage]" , "[Segment=00]" , "[Segment=01]" , "[Segment=02]" ,
#         "[Segment=03]" , "[Segment=04]" , "[Segment=05]" , "[Segment=06]" , "[Segment=07]" , "[Segment=08]" ,
#         "[Segment=09]" , "[Segment=10]" , "[Segment=11]" , "[Segment=12]" , "[Segment=13]" , "[Segment=14]" ,
#         "[Segment=15]" , "[Segment=16]" , "[Segment=17]" , "[Segment=18]" , "[Segment=19]" , "[Segment=20]" , 
#         "[Segment=21]" , "[Segment=22]" , "[Segment=23]" , "[Segment=24]" , "[Segment=25]" , "[Segment=26]" , 
#         "[Segment=27]" , "[Segment=28]" , "[Segment=29]" , "[Segment=30]" , "[Segment=XX]", "\n"]



# segment_tokens = ["[Segment=00]" , "[Segment=01]" , "[Segment=02]" ,
#         "[Segment=03]" , "[Segment=04]" , "[Segment=05]" , "[Segment=06]" , "[Segment=07]" , "[Segment=08]" ,
#         "[Segment=09]" , "[Segment=10]" , "[Segment=11]" , "[Segment=12]" , "[Segment=13]" , "[Segment=14]" ,
#         "[Segment=15]" , "[Segment=16]" , "[Segment=17]" , "[Segment=18]" , "[Segment=19]" , "[Segment=20]" , 
#         "[Segment=21]" , "[Segment=22]" , "[Segment=23]" , "[Segment=24]" , "[Segment=25]" , "[Segment=26]" , 
#         "[Segment=27]" , "[Segment=28]" , "[Segment=29]" , "[Segment=30]" , "[Segment=XX]"]


# headline_tokens = ["[H1Start]" , "[H2Start]" , "[H3Start]" , "[H4Start]" , "[H5Start]" , "[H6Start]"]



# <tr>
# <td>
# [newline]
# [h1]
# [h2]
# [h3]
# [h4]
# [h5]
# [h6]
# [imagestart]
# [imageend]
# [unorderedlist]
# [orderedlist]
# [codestart]
# [codeend]
# [linkstart]
# [linkend]
# [CLS]
# [SEP]
# [Q]
# [segment=00]
# [segment=01]
# [segment=02]
# [segment=03]
# [segment=04]
# [segment=05]
# [segment=06]
# [segment=07]
# [segment=08]
# [segment=09]
# [segment=10]
# [segment=11]
# [segment=12]
# [segment=13]
# [segment=14]
# [segment=15]
# [segment=16]
# [segment=17]
# [segment=18]
# [segment=19]
# [segment=20]
# [segment=21]
# [segment=22]
# [segment=23]
# [segment=24]
# [segment=25]
# [segment=26]
# [segment=27]
# [segment=28]
# [segment=29]
# [segment=30]
# [segment=xx]


if __name__ == "__main__":
    print(len(stopper))