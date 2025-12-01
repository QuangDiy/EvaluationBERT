# Evaluation BERT for Vietnamese

## ViGLUE

### Evaluation Results

<table style="font-family: 'Times New Roman', Times, serif;">
<thead>
<tr>
<th rowspan="2" style="text-align:center; vertical-align:middle;">Model</th>
<th rowspan="2" style="text-align:center; vertical-align:middle;">Params</th>
<th rowspan="2" style="text-align:center; vertical-align:middle;">Seed</th>
<th colspan="2" style="text-align:center;">MNLI</th>
<th style="text-align:center;">QNLI</th>
<th style="text-align:center;">RTE</th>
<th style="text-align:center;">VNRTE</th>
<th style="text-align:center;">WNLI</th>
<th style="text-align:center;">SST2</th>
<th style="text-align:center;">VSFC</th>
<th style="text-align:center;">VSMEC</th>
<th colspan="2" style="text-align:center;">MRPC</th>
<th colspan="2" style="text-align:center;">QQP</th>
<th style="text-align:center;">CoLA</th>
<th style="text-align:center;">VToC</th>
</tr>
<tr>
<th style="text-align:center;">Matched</th>
<th style="text-align:center;">Mismatched</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">F1</th>
<th style="text-align:center;">Acc.</th>
<th style="text-align:center;">F1</th>
<th style="text-align:center;">MCC</th>
<th style="text-align:center;">Acc.</th>
</tr>
</thead>
<tbody>
<tr><td colspan="18" style="font-weight: bold; text-align:center;">Baseline</td></tr>
<tr>
<td><strong>QuangDuy/modernbert-tiny-checkpoint-55000ba</strong></td>
<td style="text-align:center">34M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">33.2</td>
<td style="text-align:center">33.2</td>
<td style="text-align:center">51.2</td>
<td style="text-align:center">50.7</td>
<td style="text-align:center">53.27</td>
<td style="text-align:center">57.5</td>
<td style="text-align:center">49.1</td>
<td style="text-align:center">31.78</td>
<td style="text-align:center">12.84</td>
<td style="text-align:center">55.6</td>
<td style="text-align:center">66.8</td>
<td style="text-align:center">28.9</td>
<td style="text-align:center">29.2</td>
<td style="text-align:center">1.1</td>
<td style="text-align:center">5.84</td>
</tr>
<tr>
<td><strong>vinai/phobert-base</strong></td>
<td style="text-align:center">135M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">33.5</td>
<td style="text-align:center">33.5</td>
<td style="text-align:center">49.8</td>
<td style="text-align:center">48.6</td>
<td style="text-align:center">47.97</td>
<td style="text-align:center">37.7</td>
<td style="text-align:center">49.6</td>
<td style="text-align:center">11.27</td>
<td style="text-align:center">17.31</td>
<td style="text-align:center">41.7</td>
<td style="text-align:center">32.1</td>
<td style="text-align:center">71.1</td>
<td style="text-align:center">25.0</td>
<td style="text-align:center">-1.3</td>
<td style="text-align:center">4.15</td>
</tr>
<tr>
<td><strong>Fsoft-AIC/videberta-xsmall</strong></td>
<td style="text-align:center">70M</td>
<td style="text-align:center">42</td>
<td style="text-align:center">32.9</td>
<td style="text-align:center">32.9</td>
<td style="text-align:center">50.6</td>
<td style="text-align:center">50.7</td>
<td style="text-align:center">61.17</td>
<td style="text-align:center">46.6</td>
<td style="text-align:center">49.9</td>
<td style="text-align:center">14.81</td>
<td style="text-align:center">18.61</td>
<td style="text-align:center">43.88</td>
<td style="text-align:center">44.37</td>
<td style="text-align:center">18.8</td>
<td style="text-align:center">29.6</td>
<td style="text-align:center">-3.2</td>
<td style="text-align:center">7.09</td>
</tr>
<tr><td colspan="18" style="font-weight: bold; text-align:center;">Downstream task</td></tr>
<tr>
<td><strong>Tên_Model_SFT_Của_Bạn</strong></td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
</tr>
</tbody>
</table>

## Perplexity